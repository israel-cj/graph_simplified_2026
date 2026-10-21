"""Build a bicycle graph for Grenoble and classify every edge by traffic stress.

The classifier follows the structure of the revised Level of Traffic Stress
methodology, but it is necessarily an OSM-based approximation. OpenStreetMap
usually does not contain traffic volume, parking turnover, driveway frequency,
buffer width, or complete intersection-crossing data.

Six display classes are produced:
    LTS 0, LTS 1, LTS 2, LTS 2.5, LTS 3, and LTS 4-5.
The raw ``lts_score`` attribute still distinguishes LTS 4 from LTS 5.
Unknown edges are kept as ``Unknown`` and are not counted as a seventh class.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re
from typing import Any

import networkx as nx
import osmnx as ox
import pandas as pd


# Keep the OSM tags needed for classification. OSMnx otherwise drops many of
# these tags when it builds the graph.
REQUIRED_WAY_TAGS = {
    "access",
    "bicycle",
    "bicycle_road",
    "bus",
    "busway",
    "cycleway",
    "cycleway:both",
    "cycleway:both:buffer",
    "cycleway:both:lane",
    "cycleway:both:oneway",
    "cycleway:left",
    "cycleway:left:buffer",
    "cycleway:left:lane",
    "cycleway:left:oneway",
    "cycleway:right",
    "cycleway:right:buffer",
    "cycleway:right:lane",
    "cycleway:right:oneway",
    "cyclestreet",
    "foot",
    "footway",
    "highway",
    "is_sidepath",
    "junction",
    "lanes",
    "lit",
    "maxspeed",
    "motor_vehicle",
    "name",
    "oneway",
    "oneway:bicycle",
    "parking",
    "parking:both",
    "parking:left",
    "parking:right",
    "segregated",
    "separation",
    "shoulder",
    "sidewalk",
    "sidewalk:both:bicycle",
    "sidewalk:left:bicycle",
    "sidewalk:right:bicycle",
    "smoothness",
    "surface",
    "tracktype",
    "width",
}

# Six requested map classes. LTS 4 and 5 are merged only for display.
LTS_DISPLAY_GROUPS: dict[str, tuple[float, ...]] = {
    "LTS 0 - no traffic": (0.0,),
    "LTS 1 - very low": (1.0,),
    "LTS 2 - low": (2.0,),
    "LTS 2.5 - moderate-low": (2.5,),
    "LTS 3 - moderate-high": (3.0,),
    "LTS 4-5 - high/very high": (4.0, 5.0),
}

# Conventional stress-map colors. Unknown data is gray.
LTS_COLORS = {
    "LTS 0 - no traffic": "#006d2c",
    "LTS 1 - very low": "#31a354",
    "LTS 2 - low": "#addd8e",
    "LTS 2.5 - moderate-low": "#fee08b",
    "LTS 3 - moderate-high": "#fdae61",
    "LTS 4-5 - high/very high": "#d73027",
    "Unknown": "#969696",
}

# Default values are deliberately conservative and can be changed for a local
# Grenoble calibration. Values are km/h and total through lanes.
DEFAULT_SPEED_KMH = {
    "motorway": 110.0,
    "motorway_link": 70.0,
    "trunk": 90.0,
    "trunk_link": 70.0,
    "primary": 50.0,
    "primary_link": 50.0,
    "secondary": 50.0,
    "secondary_link": 50.0,
    "tertiary": 50.0,
    "tertiary_link": 50.0,
    "unclassified": 30.0,
    "residential": 30.0,
    "living_street": 20.0,
    "service": 20.0,
    "pedestrian": 20.0,
    "cycleway": 20.0,
    "path": 15.0,
    "footway": 10.0,
    "steps": 5.0,
}

DEFAULT_LANES = {
    "motorway": 4,
    "trunk": 4,
    "primary": 2,
    "secondary": 2,
    "tertiary": 2,
    "unclassified": 2,
    "residential": 2,
    "living_street": 1,
    "service": 1,
    "pedestrian": 1,
    "cycleway": 1,
    "path": 1,
    "footway": 1,
    "steps": 1,
}

HIGHWAY_STRESS_ORDER = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "service",
    "living_street",
    "pedestrian",
    "cycleway",
    "path",
    "footway",
    "steps",
]


# ---------------------------------------------------------------------------
# Tag parsing helpers
# ---------------------------------------------------------------------------

def _flatten(value: Any) -> list[Any]:
    """Flatten OSMnx list-valued attributes without treating strings as lists."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        output: list[Any] = []
        for item in value:
            output.extend(_flatten(item))
        return output
    return [value]


def _values(data: dict[str, Any], *keys: str) -> set[str]:
    values: set[str] = set()
    for key in keys:
        for item in _flatten(data.get(key)):
            text = str(item).strip().lower()
            if text:
                # OSMnx can aggregate values; OSM tags may also use semicolons.
                values.update(part.strip() for part in text.split(";") if part.strip())
    return values


def _has(data: dict[str, Any], keys: Iterable[str], wanted: Iterable[str]) -> bool:
    found = _values(data, *keys)
    target = {str(value).lower() for value in wanted}
    return bool(found & target)


def _has_fragment(data: dict[str, Any], keys: Iterable[str], fragments: Iterable[str]) -> bool:
    found = _values(data, *keys)
    return any(fragment in value for value in found for fragment in fragments)


def _worst_highway(data: dict[str, Any]) -> str:
    highways = _values(data, "highway")
    for highway in HIGHWAY_STRESS_ORDER:
        if highway in highways:
            return highway
    return next(iter(highways), "unclassified")


def _parse_numbers(value: Any) -> list[float]:
    numbers: list[float] = []
    for item in _flatten(value):
        text = str(item).strip().lower().replace(",", ".")
        if not text:
            continue
        for part in text.split(";"):
            match = re.search(r"\d+(?:\.\d+)?", part)
            if match:
                number = float(match.group())
                if "mph" in part:
                    number *= 1.609344
                numbers.append(number)
    return numbers


def infer_speed_kmh(data: dict[str, Any]) -> float:
    """Return the highest tagged/default speed, following weakest-link logic."""
    raw = _values(data, "maxspeed")
    coded_speeds = {
        "fr:urban": 50.0,
        "fr:rural": 80.0,
        "fr:motorway": 130.0,
        "fr:zone30": 30.0,
        "fr:living_street": 20.0,
        "walk": 10.0,
    }
    candidates = [coded_speeds[value] for value in raw if value in coded_speeds]
    candidates.extend(_parse_numbers(data.get("maxspeed")))
    if candidates:
        return max(candidates)
    return DEFAULT_SPEED_KMH.get(_worst_highway(data), 50.0)


def infer_lanes(data: dict[str, Any]) -> int:
    candidates = _parse_numbers(data.get("lanes"))
    if candidates:
        return max(1, int(round(max(candidates))))
    highway = _worst_highway(data)
    default = DEFAULT_LANES.get(highway, 2)
    if _has(data, ["oneway"], ["yes", "1", "true", "-1"]):
        return min(default, 1)
    return default


# ---------------------------------------------------------------------------
# Facility and LTS classification
# ---------------------------------------------------------------------------

def classify_facility(data: dict[str, Any]) -> str:
    highway = _worst_highway(data)
    cycleway_keys = ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both")
    cycleway_values = _values(data, *cycleway_keys)

    if highway == "steps":
        return "steps"

    if _has(data, ["cyclestreet", "bicycle_road"], ["yes", "1", "true"]):
        return "bicycle street"

    if highway == "living_street":
        return "shared street"

    if _has(data, ["is_sidepath"], ["no"]) or _has(data, ["separation"], ["independent"]):
        if highway in {"cycleway", "path", "footway"}:
            return "independent path"

    is_sidepath = (
        _has(data, ["is_sidepath"], ["yes"])
        or _has(data, ["footway"], ["sidewalk"])
        or bool(_values(data, "sidewalk:both:bicycle", "sidewalk:left:bicycle", "sidewalk:right:bicycle"))
    )

    if any(value in cycleway_values for value in {"track", "opposite_track", "separate"}):
        return "separated bike lane"

    if highway == "cycleway":
        return "sidepath/separated path" if is_sidepath else "dedicated cycleway"

    if highway in {"path", "footway"} and _has(
        data,
        ["bicycle"],
        ["yes", "designated", "official", "permissive", "destination"],
    ):
        if _has(data, ["segregated"], ["yes"]):
            return "segregated shared-use path"
        return "shared-use path"

    if highway == "pedestrian" and not _has(data, ["bicycle"], ["no"]):
        return "pedestrian zone with bicycle access"

    if any(value in cycleway_values for value in {"lane", "opposite_lane"}):
        return "painted bike lane"

    if "shoulder" in cycleway_values or _has(data, ["shoulder"], ["yes", "both", "left", "right"]):
        return "bikeable shoulder"

    if any(value in cycleway_values for value in {"shared_lane", "shared", "opposite"}):
        return "shared-lane marking"

    if any(value in cycleway_values for value in {"share_busway", "opposite_share_busway"}) or _has_fragment(
        data, ["busway"], ["lane"]
    ):
        return "shared bus lane"

    return "mixed traffic"


def _mixed_traffic_lts(speed: float, lanes: int, highway: str) -> float:
    # Metric adaptation of the revised mixed-traffic table. Because OSM rarely
    # contains ADT, center-line, and parking-turnover data, ambiguous cases are
    # assigned the more conservative (higher-stress) value.
    if highway == "living_street":
        return 1.0
    if speed <= 20 and lanes <= 2:
        return 1.0
    if speed <= 30 and lanes <= 2:
        return 2.0
    if speed <= 40 and lanes <= 3:
        return 2.5
    if speed <= 50 and lanes <= 3:
        return 3.0
    if speed <= 50:
        return 4.0
    if speed < 70:
        return 4.0
    return 5.0


def _painted_lane_lts(speed: float, lanes: int) -> float:
    # Metric adaptation of the revised bike-lane table.
    if speed <= 50:
        if lanes <= 3:
            return 2.0
        if lanes <= 5:
            return 2.5
        return 3.0
    if speed <= 60:
        return 3.0
    if speed < 70:
        return 4.0
    return 5.0


def _shoulder_lts(speed: float, lanes: int) -> float:
    # Metric adaptation of the revised bikeable-shoulder table.
    if speed <= 50:
        if lanes <= 3:
            return 2.0
        if lanes <= 5:
            return 2.5
        return 3.0
    if speed <= 60:
        return 3.0
    if speed < 70:
        return 3.0 if lanes <= 3 else 4.0
    return 4.0


def classify_lts(data: dict[str, Any]) -> tuple[float | None, str, str]:
    """Return ``(score, facility, explanation)`` for one graph edge."""
    facility = classify_facility(data)
    speed = infer_speed_kmh(data)
    lanes = infer_lanes(data)
    highway = _worst_highway(data)

    # Steps are a physical-access/rideability issue, not a traffic-stress class.
    if facility == "steps":
        return None, facility, "Steps are excluded from segment LTS; treat them as a separate impedance."

    if facility == "independent path":
        return 0.0, facility, "Explicit independent right-of-way."

    if facility in {"bicycle street", "shared street"}:
        return 1.0, facility, "Neighborhood greenway/shared-street treatment."

    if facility in {"dedicated cycleway", "segregated shared-use path"}:
        return 1.0, facility, "Dedicated or segregated facility; roadside buffer detail is unavailable in OSM."

    if facility in {"sidepath/separated path", "separated bike lane"}:
        if speed <= 60 and lanes <= 5:
            return 1.0, facility, "Separated facility on a lower-speed/moderate-width road."
        return 2.0, facility, "Separated facility, but adjacent traffic speed/width raises stress."

    if facility == "shared-use path":
        # Shared paths are separated from motor traffic but not necessarily from
        # pedestrians or driveway conflicts. Use LTS 2 unless explicitly segregated.
        return 2.0, facility, "Shared path; buffer and driveway frequency are not reliably tagged."

    if facility == "pedestrian zone with bicycle access":
        return 1.0 if speed <= 20 else 2.0, facility, "Pedestrian-priority area with bicycle access."

    if facility == "painted bike lane":
        score = _painted_lane_lts(speed, lanes)
        return score, facility, f"Painted lane with estimated {speed:g} km/h and {lanes} through lane(s)."

    if facility == "bikeable shoulder":
        score = _shoulder_lts(speed, lanes)
        return score, facility, f"Bikeable shoulder with estimated {speed:g} km/h and {lanes} through lane(s)."

    if facility == "shared-lane marking":
        score = max(2.0, _mixed_traffic_lts(speed, lanes, highway))
        return score, facility, f"Shared-lane marking; cyclist remains in mixed traffic at about {speed:g} km/h."

    if facility == "shared bus lane":
        if speed <= 30 and lanes <= 2:
            score = 2.5
        elif speed <= 50 and lanes <= 3:
            score = 3.0
        else:
            score = 4.0
        return score, facility, f"Shared bus lane with estimated {speed:g} km/h and {lanes} lane(s)."

    score = _mixed_traffic_lts(speed, lanes, highway)
    return score, facility, f"Mixed traffic with estimated {speed:g} km/h and {lanes} through lane(s)."


def display_group(score: float | None) -> str:
    if score is None:
        return "Unknown"
    for label, values in LTS_DISPLAY_GROUPS.items():
        if score in values:
            return label
    return "Unknown"


# ---------------------------------------------------------------------------
# Graph construction, subsets, and outputs
# ---------------------------------------------------------------------------

def normalize_polygon(poly: Any) -> Any:
    """Return a single WGS84 Shapely geometry from a polygon/GeoSeries/GDF."""
    if hasattr(poly, "geometry"):
        frame = poly
        if getattr(frame, "crs", None) is not None:
            try:
                frame = frame.to_crs(4326)
            except Exception as exc:
                raise ValueError("Could not transform the polygon to EPSG:4326.") from exc
        geometry = frame.geometry
        return geometry.union_all() if hasattr(geometry, "union_all") else geometry.unary_union

    if hasattr(poly, "crs") and hasattr(poly, "to_crs"):
        series = poly.to_crs(4326) if poly.crs is not None else poly
        return series.union_all() if hasattr(series, "union_all") else series.unary_union

    # A bare Shapely geometry has no CRS metadata. OSMnx expects longitude/latitude.
    return poly


def configure_osmnx() -> None:
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) | REQUIRED_WAY_TAGS)


def annotate_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    for _, _, _, data in G.edges(keys=True, data=True):
        score, facility, explanation = classify_lts(data)
        data["lts_score"] = -1.0 if score is None else float(score)
        data["lts"] = "unknown" if score is None else str(score).rstrip("0").rstrip(".")
        data["lts_group"] = display_group(score)
        data["facility_type"] = facility
        data["lts_reason"] = explanation
        data["estimated_speed_kmh"] = float(infer_speed_kmh(data))
        data["estimated_lanes"] = int(infer_lanes(data))
    return G


def largest_weak_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return the largest weakly connected component without damaging ``G``."""
    if len(G) == 0:
        return G.copy()
    if G.is_directed():
        nodes = max(nx.weakly_connected_components(G), key=len)
    else:
        nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def split_graph_by_lts(G: nx.MultiDiGraph) -> dict[str, nx.MultiDiGraph]:
    subsets: dict[str, nx.MultiDiGraph] = {}
    for label in [*LTS_DISPLAY_GROUPS, "Unknown"]:
        edge_ids = [
            (u, v, key)
            for u, v, key, data in G.edges(keys=True, data=True)
            if data.get("lts_group") == label
        ]
        subsets[label] = G.edge_subgraph(edge_ids).copy()
    return subsets


def summarize_lts(G: nx.MultiDiGraph) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in [*LTS_DISPLAY_GROUPS, "Unknown"]:
        matching = [
            data
            for _, _, _, data in G.edges(keys=True, data=True)
            if data.get("lts_group") == label
        ]
        rows.append(
            {
                "lts_group": label,
                "directed_edges": len(matching),
                "directed_length_km": sum(float(data.get("length", 0.0)) for data in matching) / 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _save_graphml(G: nx.MultiDiGraph, filepath: Path) -> None:
    if hasattr(ox, "io") and hasattr(ox.io, "save_graphml"):
        ox.io.save_graphml(G, filepath=filepath)
    else:  # OSMnx 1.x compatibility
        ox.save_graphml(G, filepath=filepath)


def plot_lts_graph(G: nx.MultiDiGraph, filepath: str | Path) -> None:
    edge_colors = [LTS_COLORS.get(data.get("lts_group", "Unknown"), "#969696") for _, _, _, data in G.edges(keys=True, data=True)]
    ox.plot_graph(
        G,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=1.2,
        bgcolor="white",
        show=False,
        close=True,
        save=True,
        filepath=str(filepath),
        dpi=250,
    )


def build_lts_graph(
    poly: Any,
    output_dir: str | Path = "grenoble_lts_output",
    *,
    remove_steps: bool = True,
) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph, dict[str, nx.MultiDiGraph], pd.DataFrame]:
    """Download Grenoble's bicycle network inside ``poly`` and classify it.

    Returns
    -------
    full_graph
        All retained components, with edge attributes ``lts_score``, ``lts_group``,
        ``facility_type``, and ``lts_reason``.
    main_graph
        Largest weakly connected component. Weak connectivity is used rather
        than strong connectivity so legal one-way bicycle links do not wrongly
        fragment the graph.
    graphs_by_lts
        Edge-induced subgraphs for the six LTS classes plus ``Unknown``.
    summary
        Edge counts and directed edge lengths by class.
    """
    configure_osmnx()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Download once. Classify after the download rather than sending dozens of
    # overlapping Overpass filters and composing duplicate graphs.
    polygon = normalize_polygon(poly)
    G = ox.graph.graph_from_polygon(
        polygon,
        network_type="bike",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    if remove_steps:
        step_edges = [
            (u, v, key)
            for u, v, key, data in G.edges(keys=True, data=True)
            if "steps" in _values(data, "highway")
        ]
        G.remove_edges_from(step_edges)
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)

    G = annotate_graph(G)
    G_main = largest_weak_component(G)
    graphs_by_lts = split_graph_by_lts(G)
    summary = summarize_lts(G)

    _save_graphml(G, output / "grenoble_lts_full.graphml")
    _save_graphml(G_main, output / "grenoble_lts_main_component.graphml")
    summary.to_csv(output / "grenoble_lts_summary.csv", index=False)
    plot_lts_graph(G, output / "grenoble_lts_map.png")

    return G, G_main, graphs_by_lts, summary


def get_grenoble_polygon() -> Any:
    """Fetch Grenoble's administrative polygon for standalone use."""
    if hasattr(ox, "geocoder"):
        gdf = ox.geocoder.geocode_to_gdf("Grenoble, Isère, France")
    else:  # OSMnx 1.x compatibility
        gdf = ox.geocode_to_gdf("Grenoble, Isère, France")
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


if __name__ == "__main__":
    # In a notebook where your polygon already exists, use:
    #     G, G_main, graphs_by_lts, summary = build_lts_graph(poly)
    polygon = get_grenoble_polygon()
    G, G_main, graphs_by_lts, summary = build_lts_graph(polygon)
    print(summary.to_string(index=False))
    print(f"Full graph: {len(G):,} nodes and {len(G.edges):,} directed edges")
    print(f"Main component: {len(G_main):,} nodes and {len(G_main.edges):,} directed edges")
