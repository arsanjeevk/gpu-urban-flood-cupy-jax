"""FastAPI job service for operational V2 forecasts."""

from __future__ import annotations

import csv
import json
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .assets import check_v1_assets
from .cli import _defaults
from .runner import run_operational
from .live import fetch_live_rainfall

PROJECT, DEFAULT_V1 = _defaults()
RUN_ROOT = Path(os.environ.get("FLOODASTRA_V2_RUNS", PROJECT / "runs")).resolve()
PREPARED = Path(os.environ.get("FLOODASTRA_V2_DATA", PROJECT / "data/operational")).resolve()
DOMAIN = PROJECT / "data/interim/domain_epsg32643.gpkg"
V1_ROOT = Path(os.environ.get("FLOODASTRA_V1_ROOT", DEFAULT_V1)).resolve()
RUN_ROOT.mkdir(parents=True, exist_ok=True)


class RunRequest(BaseModel):
    scenario: str = Field(pattern="^(historical|live)$")
    resolution_m: Literal[10, 20, 30, 50] = 20
    recession_hours: float = Field(default=2, ge=0, le=24)


class JobManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v2-gpu")
        self.futures: dict[str, Future] = {}
        self.lock = threading.Lock()
        for metadata in RUN_ROOT.glob("*.json"):
            try:
                payload = json.loads(metadata.read_text(encoding="utf-8"))
                if payload.get("status") in {"queued", "running"}:
                    payload.update(status="failed", error_type="ServiceRestarted", message="Service restarted before this run completed.")
                    metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except (OSError, ValueError):
                continue

    def submit(self, run_id: str, rainfall: Path, scenario: str, resolution: float, recession: float):
        run_dir = RUN_ROOT / run_id
        metadata = run_dir.with_suffix(".json")
        metadata.write_text(json.dumps({"run_id": run_id, "status": "queued", "scenario": scenario}), encoding="utf-8")
        def work():
            metadata.write_text(json.dumps({"run_id": run_id, "status": "running", "scenario": scenario}), encoding="utf-8")
            try:
                result = run_operational(
                    rainfall_csv=rainfall, output_dir=run_dir, prepared_dir=PREPARED,
                    v1_root=V1_ROOT, domain_path=DOMAIN, resolution_m=resolution,
                    recession_hours=recession, scenario=scenario, variant="both", require_gpu=True,
                    checkpoint=PROJECT / "data/outputs/jax_runs/residual_net_best.msgpack",
                )
                payload = {"run_id": run_id, "status": "complete", "scenario": scenario, "manifest": result}
            except Exception as exc:
                payload = {"run_id": run_id, "status": "failed", "scenario": scenario, "error_type": type(exc).__name__, "message": str(exc)}
            metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self.lock:
            self.futures[run_id] = self.executor.submit(work)


manager = JobManager()
app = FastAPI(title="FloodAstra Gurugram V2", version="0.3.0")


def _metadata(run_id: str) -> Path:
    try:
        uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(404, "Unknown run") from exc
    return RUN_ROOT / f"{run_id}.json"


@app.get("/api/health")
def health():
    status = check_v1_assets(V1_ROOT)
    status["prepared"] = (PREPARED / "asset_manifest.json").is_file()
    status["checkpoint"] = (PROJECT / "data/outputs/jax_runs/residual_net_best.msgpack").is_file()
    return status


@app.get("/api/runs")
def runs():
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(RUN_ROOT.glob("*.json"), reverse=True)]


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    path = _metadata(run_id)
    if not path.is_file(): raise HTTPException(404, "Unknown run")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/runs")
def create_run(request: RunRequest):
    run_id = str(uuid.uuid4())
    rainfall = V1_ROOT / "examples/july2025_133mm_12h.csv"
    if request.scenario == "live":
        input_dir = RUN_ROOT / "inputs"; input_dir.mkdir(exist_ok=True)
        try:
            rainfall = fetch_live_rainfall(input_dir / f"{run_id}.csv")
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc
    manager.submit(run_id, rainfall, request.scenario, request.resolution_m, request.recession_hours)
    return {"run_id": run_id, "status": "queued"}


@app.post("/api/runs/upload")
async def upload_run(file: UploadFile = File(...), resolution_m: float = 20, recession_hours: float = 2):
    if resolution_m not in {10, 20, 30, 50}:
        raise HTTPException(422, "Resolution must be 10, 20, 30, or 50 metres.")
    run_id = str(uuid.uuid4()); input_dir = RUN_ROOT / "inputs"; input_dir.mkdir(exist_ok=True)
    rainfall = input_dir / f"{run_id}.csv"
    rainfall.write_bytes(await file.read())
    manager.submit(run_id, rainfall, "upload", resolution_m, recession_hours)
    return {"run_id": run_id, "status": "queued"}


@app.delete("/api/runs/{run_id}")
def cancel_run(run_id: str):
    future = manager.futures.get(run_id)
    if future is None or not future.cancel():
        raise HTTPException(409, "Run is active or already finished and cannot be cancelled safely.")
    path = _metadata(run_id); path.write_text(json.dumps({"run_id": run_id, "status": "cancelled"}), encoding="utf-8")
    return {"run_id": run_id, "status": "cancelled"}


@app.get("/api/runs/{run_id}/artifacts/{relative_path:path}")
def artifact(run_id: str, relative_path: str):
    _metadata(run_id)
    run_dir = (RUN_ROOT / run_id).resolve(); target = (run_dir / relative_path).resolve()
    if run_dir not in target.parents or not target.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(target)


@app.get("/api/runs/{run_id}/frames/{variant}")
def frames(run_id: str, variant: str):
    if variant not in {"raw_jax", "hybrid"}:
        raise HTTPException(404, "Unknown model variant")
    _metadata(run_id)
    summary = RUN_ROOT / run_id / variant / "hourly_geojsons/hourly_geojson_summary.csv"
    if not summary.is_file():
        return []
    with summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{**row, "hour": int(row["hour"]), "actual_time_s": float(row["actual_time_s"]),
        "max_depth_m": float(row["max_depth_m"]), "wet_cells_ge_threshold": int(row["wet_cells_ge_threshold"]),
        "geojson_url": f"/api/runs/{run_id}/artifacts/{variant}/hourly_geojsons/{Path(row['geojson_file']).name}"} for row in rows]
