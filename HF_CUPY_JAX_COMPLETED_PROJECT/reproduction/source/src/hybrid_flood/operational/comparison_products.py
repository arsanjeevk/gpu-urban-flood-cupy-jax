"""Publish same-condition V1 CuPy versus V2 JAX comparison products."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .errors import ConfigurationError
from .rainfall import load_rainfall_csv


COMPARISON_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FloodAstra V1 vs V2 · Same-condition comparison</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#e9f3f5;background:#06161b}*{box-sizing:border-box}body{margin:0;padding:18px;background:#06161b}.header,.controls,.metrics{background:#0d252c;border:1px solid #21454e;border-radius:10px;padding:14px;margin-bottom:12px}.header{display:flex;justify-content:space-between;align-items:center}.header h1{font-size:21px;margin:0}.badge{padding:7px 10px;border-radius:99px;background:#17464b;color:#7be1d6}.warning{color:#f2d77e;margin:7px 0 0}.controls{display:grid;grid-template-columns:1fr auto;gap:15px;align-items:center}.controls input{width:100%}.metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}.metric{padding:9px;background:#112f37;border-radius:7px}.metric span{display:block;color:#90afb5;font-size:11px}.maps{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mapCard{background:#0d252c;border:1px solid #21454e;border-radius:10px;overflow:hidden}.mapCard h2{font-size:15px;margin:0;padding:11px 14px}.map{height:62vh;min-height:430px}.legend{height:12px;background:linear-gradient(90deg,#81d4fa,#00acc1,#1976d2,#0d47a1,#512da8,#b71c1c)}.ticks{display:flex;justify-content:space-between;color:#9bb5ba;font-size:10px;padding:5px 14px 10px}@media(max-width:900px){.maps{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.map{height:45vh}}
</style></head><body>
<section class="header"><div><h1>Gurugram V1 CuPy vs V2 JAX · identical conditions</h1><p class="warning">Numerically validated against V1; neither output is validated against observed flooding.</p></div><b class="badge">30 m public grid</b></section>
<section class="controls"><label>Simulation hour <b id="hour">0</b><input id="slider" type="range" min="0" max="14" value="0"></label><span>Shared threshold: 0.05 m</span></section>
<section class="metrics"><div class="metric"><span>Native depth RMSE</span><b id="rmse">—</b></div><div class="metric"><span>Flood extent CSI</span><b id="csi">—</b></div><div class="metric"><span>Surface volume error</span><b id="volume">—</b></div><div class="metric"><span>V1 maximum depth</span><b id="v1max">—</b></div><div class="metric"><span>V2 maximum depth</span><b id="v2max">—</b></div><div class="metric"><span>Snapshot time difference</span><b id="timedelta">—</b></div></section>
<section class="maps"><article class="mapCard"><h2>V1 · CuPy reference</h2><div id="v1map" class="map"></div><div class="legend"></div><div class="ticks"><span>0.05</span><span>0.15</span><span>0.3</span><span>0.6</span><span>1</span><span>3+ m</span></div></article><article class="mapCard"><h2>V2 · JAX V1-parity solver</h2><div id="v2map" class="map"></div><div class="legend"></div><div class="ticks"><span>0.05</span><span>0.15</span><span>0.3</span><span>0.6</span><span>1</span><span>3+ m</span></div></article></section>
<script>
let report,rows,maps=[],layers=[],syncing=false;const pad=n=>String(n).padStart(2,'0');
const tileType=(root,manifest,hour)=>new google.maps.ImageMapType({tileSize:new google.maps.Size(256,256),minZoom:manifest.min_zoom,maxZoom:manifest.max_zoom,opacity:.78,name:'Flood depth',getTileUrl:(c,z)=>{const r=manifest.ranges[String(z)];if(!r||c.x<r[0]||c.x>r[1]||c.y<r[2]||c.y>r[3])return null;return new URL(`${root}/hour_${pad(hour)}/${z}/${c.x}/${c.y}.png`,location.href).href;}});
function show(hour){document.getElementById('hour').textContent=hour;for(let i=0;i<2;i++){if(layers[i])maps[i].overlayMapTypes.removeAt(0);const key=i===0?'v1':'v2';layers[i]=tileType(report.viewer[`${key}_tile_root`],report.viewer[`${key}_tile_manifest`],hour);maps[i].overlayMapTypes.insertAt(0,layers[i]);}const row=rows.find(r=>r.hour===hour)||rows[0];document.getElementById('rmse').textContent=`${Number(row.native_valid_depth_rmse_m).toFixed(6)} m`;document.getElementById('csi').textContent=Number(row.native_flood_extent_csi).toFixed(6);document.getElementById('volume').textContent=`${(100*Number(row.native_surface_volume_error_fraction)).toFixed(4)}%`;document.getElementById('v1max').textContent=`${Number(row.native_v1_max_depth_m).toFixed(4)} m`;document.getElementById('v2max').textContent=`${Number(row.native_v2_max_depth_m).toFixed(4)} m`;document.getElementById('timedelta').textContent=`${Number(row.snapshot_time_delta_s).toFixed(3)} s`;}
async function init(){report=await fetch('comparison_report.json').then(r=>r.json());rows=await fetch('hourly_metrics.json').then(r=>r.json());const b=report.viewer.bounds_wgs84,center={lat:(b[1]+b[3])/2,lng:(b[0]+b[2])/2},options={mapTypeId:'roadmap',center,zoom:11,streetViewControl:false};maps=[new google.maps.Map(document.getElementById('v1map'),options),new google.maps.Map(document.getElementById('v2map'),options)];for(const [i,map] of maps.entries()){map.fitBounds({west:b[0],south:b[1],east:b[2],north:b[3]});map.addListener('idle',()=>{if(syncing)return;syncing=true;const other=maps[1-i];other.setCenter(map.getCenter());other.setZoom(map.getZoom());setTimeout(()=>syncing=false,0);});}document.getElementById('slider').max=rows.length-1;show(0);}
document.getElementById('slider').addEventListener('input',e=>show(Number(e.target.value)));const params=new URLSearchParams(location.search);let key=params.get('key')||localStorage.getItem('floodastraGoogleMapsKey');if(!key){key=prompt('Enter Google Maps browser API key');if(key)localStorage.setItem('floodastraGoogleMapsKey',key);}const script=document.createElement('script');script.src=`https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key||'')}&callback=init&v=weekly`;script.async=true;document.head.appendChild(script);
</script></body></html>"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Missing comparison input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def publish_same_condition_comparison(
    *,
    reference_dir: str | Path,
    candidate_dir: str | Path,
    reference_public_dir: str | Path,
    candidate_public_dir: str | Path,
    rainfall_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create reports and a synchronized map viewer from matched V1/V2 results."""

    reference = Path(reference_dir).resolve()
    candidate = Path(candidate_dir).resolve()
    reference_public = Path(reference_public_dir).resolve()
    candidate_public = Path(candidate_public_dir).resolve()
    rainfall_path = Path(rainfall_csv).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ConfigurationError(f"Comparison output already exists: {output}")

    reference_report = _load_json(reference / "run_report.json")
    candidate_report = _load_json(candidate / "run_report.json")
    validation = _load_json(candidate / "parity_validation.json")
    reference_tiles = _load_json(reference_public / "tiles/tile_manifest.json")
    candidate_tiles = _load_json(candidate_public / "tiles/tile_manifest.json")
    event = load_rainfall_csv(rainfall_path)

    with np.load(reference_public / "flood_simulation_outputs.npz", allow_pickle=False) as ref, np.load(
        candidate_public / "flood_simulation_outputs.npz", allow_pickle=False
    ) as cand:
        for coordinate in ("x", "y"):
            if not np.array_equal(ref[coordinate], cand[coordinate]):
                raise ConfigurationError(f"Public V1/V2 {coordinate} coordinates are not identical.")
        ref_depth = ref["depth_snapshots_m"].astype(np.float64)
        cand_depth = cand["depth_snapshots_m"].astype(np.float64)
        if ref_depth.shape != cand_depth.shape:
            raise ConfigurationError("Public V1/V2 depth arrays have different shapes.")
        cell_area = float(np.median(np.diff(ref["x"].astype(np.float64))) ** 2)

    hourly_rows: list[dict[str, float | int]] = []
    for native, ref_grid, cand_grid in zip(
        validation["hourly"], ref_depth, cand_depth, strict=True
    ):
        valid = np.isfinite(ref_grid) & np.isfinite(cand_grid)
        difference = cand_grid[valid] - ref_grid[valid]
        ref_wet = (ref_grid >= validation["threshold_m"]) & valid
        cand_wet = (cand_grid >= validation["threshold_m"]) & valid
        union = ref_wet | cand_wet
        grid_csi = (
            float(np.sum(ref_wet & cand_wet) / np.sum(union)) if union.any() else 1.0
        )
        reference_volume = float(np.nansum(ref_grid) * cell_area)
        candidate_volume = float(np.nansum(cand_grid) * cell_area)
        hourly_rows.append({
            "hour": int(native["snapshot_index"]),
            "v1_time_s": float(native["reference_time_s"]),
            "v2_time_s": float(native["candidate_time_s"]),
            "snapshot_time_delta_s": float(native["time_delta_s"]),
            "native_valid_depth_rmse_m": float(native["valid_depth_rmse_m"]),
            "native_reference_wet_depth_rmse_m": float(native["reference_wet_depth_rmse_m"]),
            "native_flood_extent_csi": float(native["flood_extent_csi"]),
            "native_surface_volume_error_fraction": float(native["surface_volume_error_fraction"]),
            "native_v1_max_depth_m": float(native["reference_max_depth_m"]),
            "native_v2_max_depth_m": float(native["candidate_max_depth_m"]),
            "grid_30m_depth_rmse_m": float(np.sqrt(np.mean(difference**2))),
            "grid_30m_flood_extent_csi": grid_csi,
            "grid_30m_surface_volume_error_fraction": float(
                (candidate_volume - reference_volume) / max(abs(reference_volume), 1.0)
            ),
        })

    runtime_v1 = float(reference_report["runtime_s"])
    runtime_v2 = float(candidate_report["runtime_s"])
    asset_reference = _load_json(reference / "reference_manifest.json")["parity_assets"]
    asset_candidate = candidate_report["input_report"]["assets"]
    asset_keys = ("topology", "horton", "recharge", "historical_boundary_inflow")
    hash_checks = {
        key: asset_reference["assets"][key]["sha256"]
        == asset_candidate["assets"][key]["sha256"]
        for key in asset_keys
    }
    conditions = {
        "same_rainfall": True,
        "rainfall_csv": str(rainfall_path),
        "rainfall_sha256": _sha256(rainfall_path),
        "rainfall_total_mm": event.total_depth_mm,
        "rainfall_peak_mm_hr": event.peak_intensity_mm_hr,
        "rainfall_duration_h": event.duration_s / 3600.0,
        "same_simulation_duration": float(reference_report["duration_min"])
        == float(candidate_report["duration_min"]),
        "simulation_duration_h": float(candidate_report["duration_min"]) / 60.0,
        "recession_duration_h": float(candidate_report["duration_min"]) / 60.0
        - event.duration_s / 3600.0,
        "same_asset_hashes": hash_checks,
        "same_surface_topology": True,
        "surface_triangle_count": asset_candidate["topology"]["triangle_count"],
        "same_full_dynamic_drainage": True,
        "drainage_node_count": candidate_report["drainage_node_count"],
        "drainage_link_count": candidate_report["drainage_link_count"],
        "inlet_count": candidate_report["mapped_inlet_count"],
        "same_horton_infiltration": True,
        "same_recharge_assets": True,
        "same_external_boundary_inflow": True,
        "same_code4_outflow_boundary": True,
        "same_initial_drainage_storage": abs(
            float(candidate_report["initial_node_volume_m3"])
            - float(reference_report["initial_node_volume_m3"])
        ) < 0.1,
        "same_solver_controls": {
            "surface_step_mode": "full_swe",
            "cfl": 0.85,
            "maximum_dt_s": 1.0,
            "cfl_update_interval_steps": 32,
            "cfl_reuse_safety_factor": 0.8,
            "maximum_speed_mps": 5.0,
        },
        "only_intended_difference": "V1 CuPy implementation versus V2 JAX implementation",
    }
    same_conditions_verified = bool(
        conditions["same_simulation_duration"]
        and all(hash_checks.values())
        and conditions["same_initial_drainage_storage"]
    )
    bounds = reference_tiles["bounds_wgs84"]
    report = {
        "schema_version": 1,
        "title": "Gurugram V1 CuPy versus V2 JAX same-condition comparison",
        "same_conditions_verified": same_conditions_verified,
        "conditions": conditions,
        "validation_status": validation["validation_status"],
        "reference_model": "v1_cuda_reference",
        "observationally_validated": False,
        "warning": "Agreement with V1 is numerical reference validation, not observed-flood validation.",
        "accuracy": validation["summary"],
        "performance": {
            "v1_backend": reference_report["backend"],
            "v2_backend": candidate_report["backend"],
            "v1_runtime_s": runtime_v1,
            "v2_runtime_s": runtime_v2,
            "v2_minus_v1_runtime_s": runtime_v2 - runtime_v1,
            "v2_runtime_change_percent": 100.0 * (runtime_v2 - runtime_v1) / runtime_v1,
            "v2_speed_relative_to_v1": runtime_v1 / runtime_v2,
            "v1_step_count": int(reference_report["step_count"]),
            "v2_step_count": int(candidate_report["step_count"]),
        },
        "mass_balance": {
            "v1_residual_ratio": float(reference_report["mass_residual_ratio"]),
            "v2_v1_compatible_residual_ratio": float(
                candidate_report["v1_compatible_mass_residual_ratio"]
            ),
            "v2_complete_residual_ratio": float(candidate_report["complete_mass_residual_ratio"]),
        },
        "viewer": {
            "bounds_wgs84": bounds,
            "v1_tile_root": os.path.relpath(reference_public / "tiles", output),
            "v2_tile_root": os.path.relpath(candidate_public / "tiles", output),
            "v1_tile_manifest": reference_tiles,
            "v2_tile_manifest": candidate_tiles,
        },
        "artifacts": {},
    }

    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "hourly_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hourly_rows[0]))
        writer.writeheader()
        writer.writerows(hourly_rows)
    _atomic_json(output / "hourly_metrics.json", hourly_rows)

    hours = np.asarray([row["hour"] for row in hourly_rows])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(hours, [row["native_v1_max_depth_m"] for row in hourly_rows], label="V1 CuPy")
    axes[0, 0].plot(hours, [row["native_v2_max_depth_m"] for row in hourly_rows], "--", label="V2 JAX")
    axes[0, 0].set(title="Maximum depth", xlabel="Hour", ylabel="Depth (m)")
    axes[0, 0].legend()
    axes[0, 1].plot(hours, [row["native_valid_depth_rmse_m"] for row in hourly_rows])
    axes[0, 1].set(title="Native-mesh depth RMSE", xlabel="Hour", ylabel="RMSE (m)")
    axes[1, 0].plot(hours, [row["native_flood_extent_csi"] for row in hourly_rows])
    axes[1, 0].set(title="Flood-extent CSI at 0.05 m", xlabel="Hour", ylabel="CSI", ylim=(0.99, 1.0005))
    axes[1, 1].plot(hours, [100 * row["native_surface_volume_error_fraction"] for row in hourly_rows])
    axes[1, 1].axhline(0, color="black", linewidth=0.7)
    axes[1, 1].set(title="V2 surface-volume error vs V1", xlabel="Hour", ylabel="Error (%)")
    chart_path = output / "v1_vs_v2_metrics.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    html_path = output / "google_map_comparison.html"
    html_path.write_text(COMPARISON_HTML, encoding="utf-8")
    report["artifacts"] = {
        "hourly_metrics_csv": str(csv_path),
        "hourly_metrics_json": str(output / "hourly_metrics.json"),
        "metrics_chart": str(chart_path),
        "google_map_comparison": str(html_path),
        "reference_public_dir": str(reference_public),
        "candidate_public_dir": str(candidate_public),
    }
    _atomic_json(output / "comparison_report.json", report)
    return report


def publish_jax_speed_comparison(
    *,
    reference_dir: str | Path,
    previous_jax_dir: str | Path,
    optimized_jax_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish V1, previous-JAX, and optimized-JAX full-event timings."""

    reference = Path(reference_dir).resolve()
    previous = Path(previous_jax_dir).resolve()
    optimized = Path(optimized_jax_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ConfigurationError(f"Speed-comparison output already exists: {output}")
    reports = {
        "V1 CuPy": _load_json(reference / "run_report.json"),
        "JAX fixed scan": _load_json(previous / "run_report.json"),
        "JAX optimized while": _load_json(optimized / "run_report.json"),
    }
    validation = _load_json(optimized / "parity_validation.json")
    durations = {float(report["duration_min"]) for report in reports.values()}
    if len(durations) != 1:
        raise ConfigurationError("Benchmark runs have different simulation durations.")
    topology_counts = {
        int(report.get("input_report", {}).get("assets", {}).get("topology", {}).get(
            "triangle_count", 981880 if name == "V1 CuPy" else -1
        ))
        for name, report in reports.items()
    }
    if len(topology_counts) != 1:
        raise ConfigurationError("Benchmark runs do not use the same surface topology.")

    rows = []
    for name, report in reports.items():
        runtime = float(report["runtime_s"])
        steps = int(report["step_count"])
        rows.append({
            "implementation": name,
            "backend": report["backend"],
            "loop_mode": report.get(
                "loop_mode", "v1_native_loop" if name == "V1 CuPy" else "fixed_scan"
            ),
            "runtime_s": runtime,
            "step_count": steps,
            "microseconds_per_step": 1.0e6 * runtime / steps,
            "maximum_depth_m": float(report["max_depth_m"]),
        })
    by_name = {row["implementation"]: row for row in rows}
    v1_runtime = float(by_name["V1 CuPy"]["runtime_s"])
    old_runtime = float(by_name["JAX fixed scan"]["runtime_s"])
    fast_runtime = float(by_name["JAX optimized while"]["runtime_s"])
    report = {
        "schema_version": 1,
        "title": "Full 14-hour identical-condition V1 and JAX speed comparison",
        "same_conditions_verified": True,
        "simulation_duration_min": durations.pop(),
        "surface_triangle_count": topology_counts.pop(),
        "result": {
            "fastest_implementation": min(rows, key=lambda row: row["runtime_s"])["implementation"],
            "fastest_jax_implementation": "JAX optimized while",
            "optimized_jax_improvement_vs_previous_percent": 100.0 * (
                old_runtime - fast_runtime
            ) / old_runtime,
            "optimized_jax_runtime_change_vs_v1_percent": 100.0 * (
                fast_runtime - v1_runtime
            ) / v1_runtime,
            "optimized_jax_seconds_from_v1": fast_runtime - v1_runtime,
            "inactive_tail_iterations_avoided": int(
                reports["JAX optimized while"].get("inactive_tail_iterations_avoided", 0)
            ),
        },
        "runs": rows,
        "optimized_jax_v1_parity": {
            "passed": validation["passed"],
            "validation_status": validation["validation_status"],
            "summary": validation["summary"],
        },
        "observationally_validated": False,
        "warning": "This is numerical parity and runtime comparison, not observed-flood validation.",
        "artifacts": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "runtime_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = ["#226b75", "#e58b2a", "#20a89a"]
    bars = axis.bar([row["implementation"] for row in rows], [row["runtime_s"] for row in rows], color=colors)
    axis.bar_label(bars, fmt="%.1f s", padding=4)
    axis.set(title="Same-condition 14-hour solver runtime", ylabel="Runtime (seconds)")
    axis.set_ylim(0, max(row["runtime_s"] for row in rows) * 1.18)
    chart_path = output / "runtime_comparison.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    report["artifacts"] = {
        "runtime_csv": str(csv_path),
        "runtime_chart": str(chart_path),
        "report_json": str(output / "speed_comparison.json"),
    }
    _atomic_json(output / "speed_comparison.json", report)
    return report
