"""Historical OSM snapshot wrapper for grenoble_lts_graph.py."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import osmnx as ox

from grenoble_lts_graph import build_lts_graph


def normalize_snapshot_date(snapshot_date: str | datetime) -> str:
    """Return an ISO-8601 UTC timestamp accepted by Overpass."""
    if isinstance(snapshot_date, datetime):
        value = snapshot_date
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    text = str(snapshot_date).strip()

    if len(text) == 4 and text.isdigit():
        text = f"{text}-12-31T23:59:59Z"
    elif len(text) == 10:
        text = f"{text}T23:59:59Z"

    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_historical_lts_graph(
    poly: Any,
    snapshot_date: str | datetime,
    output_dir: str | Path,
    *,
    remove_steps: bool = True,
    requests_timeout: int = 600,
    overpass_url: str | None = None,
    cache_root: str | Path = "cache_historical_osm",
):
    """Build an LTS graph from the OSM database state at one past date."""
    date_iso = normalize_snapshot_date(snapshot_date)
    date_slug = date_iso[:10]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cache_folder = Path(cache_root) / date_slug
    cache_folder.mkdir(parents=True, exist_ok=True)

    old_settings = {
        "overpass_settings": ox.settings.overpass_settings,
        "requests_timeout": ox.settings.requests_timeout,
        "cache_folder": ox.settings.cache_folder,
        "use_cache": ox.settings.use_cache,
        "overpass_url": ox.settings.overpass_url,
    }

    try:
        ox.settings.requests_timeout = int(requests_timeout)
        ox.settings.overpass_settings = (
            f'[out:json][timeout:{{timeout}}]'
            f'[date:"{date_iso}"]'
            f'{{maxsize}}'
        )
        ox.settings.cache_folder = str(cache_folder)
        ox.settings.use_cache = True

        if overpass_url is not None:
            ox.settings.overpass_url = overpass_url.rstrip("/")

        result = build_lts_graph(
            poly,
            output_dir=output,
            remove_steps=remove_steps,
        )

        (output / "snapshot_date.txt").write_text(
            date_iso + "\n",
            encoding="utf-8",
        )

        return result

    finally:
        ox.settings.overpass_settings = old_settings["overpass_settings"]
        ox.settings.requests_timeout = old_settings["requests_timeout"]
        ox.settings.cache_folder = old_settings["cache_folder"]
        ox.settings.use_cache = old_settings["use_cache"]
        ox.settings.overpass_url = old_settings["overpass_url"]


def build_year_end_snapshots(
    poly: Any,
    years: list[int],
    output_root: str | Path = "grenoble_lts_output",
    **kwargs,
):
    """Build multiple year-end snapshots with the same LTS classifier."""
    results = {}

    for year in years:
        results[year] = build_historical_lts_graph(
            poly,
            snapshot_date=f"{year}-12-31T23:59:59Z",
            output_dir=Path(output_root) / str(year),
            **kwargs,
        )

    return results
