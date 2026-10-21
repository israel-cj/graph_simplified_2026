"""Only NEW work for 2026: live/current OSM cycling + missing car roads.
No Matplotlib, no changes to the original 2017-2022 classification.
"""
from pathlib import Path
import pickle
from contextlib import contextmanager
import networkx as nx
import osmnx as ox

from grenoble_lts_graph import (
    configure_osmnx, annotate_graph, largest_weak_component,
    normalize_polygon, _values, infer_speed_kmh, infer_lanes,
)
from grenoble_workflow_fixed import prepare_bike, osm_way_ids
from grenoble_simplification_exact import simplify_lts_graph
from overpass_robust import download_graph

ROOT = Path('grenoble_lts_output/historical')

@contextmanager
def current_osm_settings(cache_folder='cache_grenoble_2026_current', timeout=600, overpass_url=None):
    keys = ('overpass_settings', 'requests_timeout', 'cache_folder', 'use_cache', 'overpass_url', 'log_console')
    old = {k: getattr(ox.settings, k) for k in keys}
    try:
        # No [date: ...]: live OSM snapshot at the time of first successful download.
        ox.settings.overpass_settings = '[out:json][timeout:{timeout}]{maxsize}'
        ox.settings.requests_timeout = int(timeout)
        ox.settings.cache_folder = str(cache_folder)
        ox.settings.use_cache = True
        ox.settings.log_console = True
        if overpass_url is not None:
            ox.settings.overpass_url = overpass_url.rstrip('/')
        print('[2026] Overpass endpoint:', ox.settings.overpass_url, flush=True)
        print('[2026] HTTP timeout:', ox.settings.requests_timeout, 'seconds', flush=True)
        yield
    finally:
        for k, v in old.items():
            setattr(ox.settings, k, v)


def _missing_car_network(bike, car, match='endpoints'):
    """match='endpoints' (original rule) or 'way' (skip car edges whose OSM ways
    all exist in the bike graph, regardless of where OSMnx split them)."""
    combined = bike.copy()
    bike_way_ids = set().union(*(osm_way_ids(d) for _,_,_,d in bike.edges(keys=True,data=True))) if bike.number_of_edges() else set()
    bike_pairs = {frozenset((u,v)) for u,v,_,_ in bike.edges(keys=True,data=True)}
    bike_ways = {(w, frozenset((u,v))) for u,v,_,d in bike.edges(keys=True,data=True) for w in osm_way_ids(d)}
    added=0
    for u,v,_,data in car.edges(keys=True,data=True):
        pair=frozenset((u,v))
        if match == 'way' and osm_way_ids(data) and osm_way_ids(data) <= bike_way_ids:
            continue
        if pair in bike_pairs or any((w,pair) in bike_ways for w in osm_way_ids(data)):
            continue
        highway = _values(data,'highway')
        speed = float(infer_speed_kmh(data)); lanes = int(infer_lanes(data))
        prohibited = bool(highway & {'motorway','motorway_link','trunk','trunk_link'}) or bool(_values(data,'bicycle') & {'no','private'})
        dangerous = speed>=50 or lanes>=4 or bool(highway & {'primary','primary_link','secondary','secondary_link','tertiary_link'})
        mandatory = bool(dangerous and not prohibited)
        for node in (u,v):
            if node not in combined: combined.add_node(node,**dict(car.nodes[node]))
        attrs = dict(data)
        attrs.update(key=5,lts=5, network_layer='missing_car',
            distance=float(data.get('length',data.get('distance',0.0))),
            estimated_speed_kmh=speed,estimated_lanes=lanes,
            bicycle_prohibited=prohibited,high_stress_car_road=dangerous,
            mandatory_upgrade_to_lts1=mandatory,
            target_lts=1 if mandatory else None,
            optimization_candidate=not prohibited,
            upgrade_reason=('independent facility required' if prohibited else 'high speed/lanes/class' if mandatory else 'no mandatory upgrade under rule'))
        # add_edge key is a networkx multigraph positional argument; remove it
        attrs.pop('key',None)
        new_key=combined.add_edge(u,v,**attrs)
        combined[u][v][new_key]['key']=5
        added+=1
    combined.graph.update(year=2026, snapshot_mode='current/live OSM', lts5_meaning='missing car-road planning layer')
    print(f'[2026] Added {added:,} directed missing-car edges (matching by endpoint/OSM way is approximate).')
    return combined


def build_2026_combined(poly=None, root=ROOT, rebuild=False, timeout=600,
                        cache_folder='cache_grenoble_2026_current', overpass_url=None,
                        match='endpoints'):
    """Reuse a full combined checkpoint; otherwise query current OSM, then simplify together."""
    folder=Path(root)/'2026';folder.mkdir(parents=True, exist_ok=True)
    prepared=folder/'grenoble_lts5_2026_pre_simplification.pkl'
    bike_file=folder/'grenoble_lts_2026_bike_prepared.pkl'
    car_file=folder/'grenoble_lts_2026_car_raw.pkl'
    simple=folder/'grenoble_lts5_2026_SIMPLE_optimization.pkl'
    if simple.exists() and not rebuild:
        with simple.open('rb') as f:return pickle.load(f)
    if prepared.exists() and not rebuild:
        with prepared.open('rb') as f: combined=pickle.load(f)
    else:
        if poly is None: raise ValueError('Supply poly to download missing 2026 networks.')
        configure_osmnx()
        polygon=normalize_polygon(poly)
        # overpass_url given -> try it first, then the other mirrors
        from overpass_robust import OVERPASS_ENDPOINTS
        endpoints = ([overpass_url] if overpass_url else []) + [e for e in OVERPASS_ENDPOINTS if e != (overpass_url or '').rstrip('/')]
        with current_osm_settings(cache_folder, timeout, overpass_url):
            if bike_file.exists() and not rebuild:
                with bike_file.open('rb') as f: bike=pickle.load(f)
            else:
                print('[2026] Downloading CURRENT bicycle network...', flush=True)
                bike_raw=download_graph(polygon,'bike',server_timeout=timeout,endpoints=endpoints,cache_folder=cache_folder)
                to_remove=[(u,v,k) for u,v,k,d in bike_raw.edges(keys=True,data=True) if 'steps' in _values(d,'highway')]
                bike_raw.remove_edges_from(to_remove)
                bike_raw.remove_nodes_from(list(nx.isolates(bike_raw)))
                bike_raw=annotate_graph(bike_raw)
                bike=prepare_bike(largest_weak_component(bike_raw))
                with bike_file.open('wb') as f:pickle.dump(bike,f,pickle.HIGHEST_PROTOCOL)
                print('[2026] Saved bicycle checkpoint:',bike_file)
            if car_file.exists() and not rebuild:
                print('[2026] Loading cached car graph:', car_file, flush=True)
                with car_file.open('rb') as f: cars=pickle.load(f)
            else:
                print('[2026] Downloading CURRENT car network...', flush=True)
                cars=download_graph(polygon,'drive',server_timeout=timeout,endpoints=endpoints,cache_folder=cache_folder)
                with car_file.open('wb') as f: pickle.dump(cars,f,pickle.HIGHEST_PROTOCOL)
                print('[2026] Saved car checkpoint:', car_file, flush=True)
        combined=_missing_car_network(bike,cars,match=match)
        with prepared.open('wb') as f:pickle.dump(combined,f,pickle.HIGHEST_PROTOCOL)
        print('[2026] Saved combined checkpoint:', prepared)
    G=simplify_lts_graph(combined)
    with simple.open('wb') as f:pickle.dump(G,f,pickle.HIGHEST_PROTOCOL)
    print('[2026] Saved simplified combined graph:',simple)
    return G
