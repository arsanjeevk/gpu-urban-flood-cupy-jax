"""Generate transparent Web-Mercator XYZ tiles from V1-compatible NPZ output."""

from __future__ import annotations

import json
from math import floor
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

from .errors import ConfigurationError

WEB_MERCATOR_HALF_WORLD = 20_037_508.342789244
TILE_SIZE = 256

GOOGLE_VIEWER = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FloodAstra 30 m Google Tiles</title><style>
html,body,#map{height:100%;margin:0;font:14px system-ui;background:#07161b}.panel{position:absolute;z-index:5;top:16px;left:16px;width:310px;padding:15px;border-radius:10px;background:#0d2329ee;color:#e8f2f4;box-shadow:0 8px 30px #0008}.panel h1{font-size:17px;margin:0 0 10px}.panel input{width:100%}.row{display:flex;justify-content:space-between;margin-top:8px}.warning{color:#f3d87d;font-size:12px}.legend{height:12px;margin-top:10px;background:linear-gradient(90deg,#81d4fa,#00acc1,#1976d2,#0d47a1,#512da8,#b71c1c)}.ticks{display:flex;justify-content:space-between;font-size:10px;color:#abc}</style></head>
<body><div id="map"></div><section class="panel"><h1>Gurugram flood depth · <span id="resolution">30</span> m tiles</h1><label>Simulation hour <b id="hour">0</b><input id="slider" type="range" min="0" max="14" value="0"></label><div class="row"><span>Layer</span><b id="model">Flood result</b></div><div class="legend"></div><div class="ticks"><span>0.05</span><span>0.15</span><span>0.3</span><span>0.6</span><span>1</span><span>3+ m</span></div><p class="warning" id="warning">Not observationally validated.</p></section>
<script>
let map,manifest,overlay;const root=new URL('.',location.href);const pad=n=>String(n).padStart(2,'0');
async function init(){manifest=await fetch('tile_manifest.json').then(r=>r.json());document.getElementById('model').textContent=manifest.model_variant||'Flood result';document.getElementById('resolution').textContent=manifest.grid_resolution_m||30;document.getElementById('slider').max=Math.max(...manifest.hours);document.getElementById('warning').textContent=manifest.drainage_mode==='v1_full_dynamic_network'?'Full V1 drainage parity; validated against V1, not observed flooding.':'Simplified drainage; not observationally validated.';const b=manifest.bounds_wgs84;map=new google.maps.Map(document.getElementById('map'),{mapTypeId:'roadmap',center:{lat:(b[1]+b[3])/2,lng:(b[0]+b[2])/2},zoom:11,streetViewControl:false});map.fitBounds({west:b[0],south:b[1],east:b[2],north:b[3]});show(0);}
function show(hour){if(overlay)map.overlayMapTypes.removeAt(0);overlay=new google.maps.ImageMapType({tileSize:new google.maps.Size(256,256),minZoom:manifest.min_zoom,maxZoom:manifest.max_zoom,opacity:.78,name:'Flood depth',getTileUrl:(c,z)=>{const r=manifest.ranges[String(z)];if(!r||c.x<r[0]||c.x>r[1]||c.y<r[2]||c.y>r[3])return null;return new URL(`hour_${pad(hour)}/${z}/${c.x}/${c.y}.png`,root).href;}});map.overlayMapTypes.insertAt(0,overlay);}
document.getElementById('slider').addEventListener('input',e=>{document.getElementById('hour').textContent=e.target.value;show(Number(e.target.value));});
const params=new URLSearchParams(location.search);let key=params.get('key')||localStorage.getItem('floodastraGoogleMapsKey');if(!key){key=prompt('Enter Google Maps browser API key');if(key)localStorage.setItem('floodastraGoogleMapsKey',key);}const script=document.createElement('script');script.src=`https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key||'')}&callback=init&v=weekly`;script.async=true;document.head.appendChild(script);
</script></body></html>"""


def write_google_viewer(tile_root: str | Path) -> Path:
    target = Path(tile_root) / "google_map.html"
    target.write_text(GOOGLE_VIEWER, encoding="utf-8")
    return target


def _tile_range(bounds: tuple[float, float, float, float], zoom: int) -> tuple[int, int, int, int]:
    west, south, east, north = bounds
    count = 1 << zoom
    span = 2 * WEB_MERCATOR_HALF_WORLD / count
    min_x = max(0, floor((west + WEB_MERCATOR_HALF_WORLD) / span))
    max_x = min(count - 1, floor((east + WEB_MERCATOR_HALF_WORLD) / span))
    min_y = max(0, floor((WEB_MERCATOR_HALF_WORLD - north) / span))
    max_y = min(count - 1, floor((WEB_MERCATOR_HALF_WORLD - south) / span))
    return min_x, max_x, min_y, max_y


def _tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    span = 2 * WEB_MERCATOR_HALF_WORLD / (1 << zoom)
    west = -WEB_MERCATOR_HALF_WORLD + x * span
    east = west + span
    north = WEB_MERCATOR_HALF_WORLD - y * span
    south = north - span
    return west, south, east, north


def _colorize(depth: np.ndarray, threshold_m: float) -> np.ndarray:
    rgba = np.zeros((*depth.shape, 4), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth >= threshold_m)
    stops = (
        (0.05, (129, 212, 250, 175)),
        (0.15, (0, 172, 193, 195)),
        (0.30, (25, 118, 210, 205)),
        (0.60, (13, 71, 161, 215)),
        (1.00, (81, 45, 168, 225)),
        (3.00, (183, 28, 28, 235)),
    )
    for lower, color in stops:
        rgba[valid & (depth >= lower)] = color
    return rgba


def generate_xyz_tiles(
    run_variant_dir: str | Path,
    *,
    min_zoom: int = 9,
    max_zoom: int = 15,
    threshold_m: float = 0.05,
) -> Path:
    variant_dir = Path(run_variant_dir).resolve()
    archive_path = variant_dir / "flood_simulation_outputs.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing V1-compatible archive: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        depth = archive["depth_snapshots_m"].astype(np.float32)
        times = archive["snapshot_times_s"].astype(np.float64)
        x = archive["x"].astype(np.float64)
        y = archive["y"].astype(np.float64)
    if depth.shape[1:] != (len(y), len(x)):
        raise ValueError("Depth snapshots do not match x/y coordinates.")
    resolution = float(np.median(np.diff(x)))
    report_path = variant_dir / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    source_transform = from_origin(
        x[0] - resolution / 2,
        y[-1] + resolution / 2,
        resolution,
        resolution,
    )
    source_bounds = (
        x[0] - resolution / 2,
        y[0] - resolution / 2,
        x[-1] + resolution / 2,
        y[-1] + resolution / 2,
    )
    mercator_bounds = transform_bounds("EPSG:32643", "EPSG:3857", *source_bounds)
    lonlat_bounds = transform_bounds("EPSG:32643", "EPSG:4326", *source_bounds)
    tile_root = variant_dir / "tiles"
    if tile_root.exists():
        raise ConfigurationError(f"Tile output already exists and will not be overwritten: {tile_root}")
    tile_root.mkdir(parents=True, exist_ok=False)
    ranges = {}
    for zoom in range(min_zoom, max_zoom + 1):
        ranges[str(zoom)] = list(_tile_range(mercator_bounds, zoom))
    hours = []
    tile_count = 0
    for snapshot_index, time_s in enumerate(times):
        hour = int(round(time_s / 3600.0))
        hours.append(hour)
        source = np.flipud(depth[snapshot_index])
        for zoom in range(min_zoom, max_zoom + 1):
            min_x, max_x, min_y, max_y = ranges[str(zoom)]
            for tile_x in range(min_x, max_x + 1):
                for tile_y in range(min_y, max_y + 1):
                    bounds = _tile_bounds(tile_x, tile_y, zoom)
                    destination = np.full((TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float32)
                    reproject(
                        source=source,
                        destination=destination,
                        src_transform=source_transform,
                        src_crs="EPSG:32643",
                        src_nodata=np.nan,
                        dst_transform=from_bounds(*bounds, TILE_SIZE, TILE_SIZE),
                        dst_crs="EPSG:3857",
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                    rgba = _colorize(destination, threshold_m)
                    if not rgba[..., 3].any():
                        continue
                    target = tile_root / f"hour_{hour:02d}" / str(zoom) / str(tile_x) / f"{tile_y}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgba, mode="RGBA").save(target, "PNG", optimize=True)
                    tile_count += 1
    manifest = {
        "schema_version": 1,
        "tile_scheme": "xyz",
        "tile_size": TILE_SIZE,
        "format": "png",
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "hours": hours,
        "threshold_m": threshold_m,
        "grid_resolution_m": resolution,
        "model_variant": report.get("model_variant", "unknown"),
        "validation_status": report.get("validation_status", "unvalidated"),
        "drainage_mode": report.get("drainage_mode", "unknown"),
        "observationally_validated": bool(report.get("observationally_validated", False)),
        "bounds_wgs84": list(lonlat_bounds),
        "ranges": ranges,
        "url_template": "hour_{hour}/{z}/{x}/{y}.png",
        "tile_count": tile_count,
    }
    manifest_path = tile_root / "tile_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_google_viewer(tile_root)
    comparison_path = variant_dir.parent / "comparison_manifest.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        public_key = f"public_grid_{resolution:g}m"
        public = comparison.setdefault("candidate_artifacts", {}).setdefault(
            public_key, {"output_dir": str(variant_dir)}
        )
        public["tiles"] = {
            "manifest": str(manifest_path),
            "google_map_html": str(tile_root / "google_map.html"),
            "tile_count": tile_count,
        }
        temporary = comparison_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(comparison_path)
    return manifest_path
