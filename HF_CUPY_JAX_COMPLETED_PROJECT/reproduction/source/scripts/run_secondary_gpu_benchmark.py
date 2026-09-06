#!/usr/bin/env python3
"""3 warm-ups + 10 measured GPU runs on an identical 60-minute production case."""
from __future__ import annotations
import csv,json,math,shutil,statistics,subprocess,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"completed_work/secondary_benchmark"; RAW=OUT/"raw"
RAIN=ROOT/"examples/user_24h_rainfall.csv"; CFG=ROOT/"configs/production_parity_gpu.yaml"

def run(cmd,log):
 t=time.perf_counter(); p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); wall=time.perf_counter()-t; log.write_text(p.stdout+p.stderr); return p.returncode,wall
def correctness(candidate,baseline):
 with np.load(candidate/"flood_simulation_outputs.npz") as a,np.load(baseline/"flood_simulation_outputs.npz") as b:
  valid=a["active_surface_mask"].astype(bool)&~a["building_mask"].astype(bool); x=a["depth_snapshots_m"][:,valid].astype(np.float64); y=b["depth_snapshots_m"][:,valid].astype(np.float64); d=x-y; aw=x>=.05; bw=y>=.05; csi=np.min((aw&bw).sum(axis=1)/np.maximum((aw|bw).sum(axis=1),1)); return bool(np.isfinite(x).all() and x.min()>=-1e-7 and np.sqrt(np.mean(d*d))<=.05 and csi>=.95),float(np.sqrt(np.mean(d*d))),float(csi)
def stats(v):
 a=np.asarray(v,float); n=len(a); sd=float(a.std()); se=sd/math.sqrt(n); return {"n":n,"mean_s":float(a.mean()),"median_s":float(np.median(a)),"minimum_s":float(a.min()),"maximum_s":float(a.max()),"population_std_s":sd,"iqr_s":float(np.percentile(a,75)-np.percentile(a,25)),"coefficient_of_variation":sd/float(a.mean()),"confidence_interval_95_s":[float(a.mean()-1.96*se),float(a.mean()+1.96*se)]}
def main():
 OUT.mkdir(parents=True,exist_ok=True); RAW.mkdir(exist_ok=True); records=[]; baselines={}
 for backend in ("cupy","jax"):
  for i in range(13):
   phase="warmup" if i<3 else "measured"; idx=i+1 if i<3 else i-2; dest=RAW/f"{backend}_{phase}_{idx:02d}"; log=RAW/f"{backend}_{phase}_{idx:02d}.log"
   if dest.exists(): raise SystemExit(f"refusing existing {dest}")
   if backend=="cupy": cmd=[str(ROOT/".venv/bin/python"),"-m","gurugram_flood.cli","run","--rainfall",str(RAIN),"--config",str(CFG),"--output-dir",str(dest)]
   else: cmd=[str(ROOT/".venv/bin/python"),"-m","hybrid_flood.production_gpu","run-jax","--rainfall",str(RAIN),"--output-dir",str(dest),"--duration-min","60","--root",str(ROOT),"--loop-mode","while"]
   rc,wall=run(cmd,log)
   if rc: raise SystemExit(f"{backend} {phase} {idx} failed; see {log}")
   rep=json.loads((dest/"run_report.json").read_text()); baselines.setdefault(backend,dest); ok,rmse,csi=correctness(dest,baselines[backend])
   records.append({"backend":backend,"phase":phase,"repetition":idx,"solver_runtime_s":rep["runtime_s"],"wall_pipeline_s":wall,"step_count":rep["step_count"],"steps_per_s":rep["step_count"]/rep["runtime_s"],"simulated_hours_per_wall_hour":3600/rep["runtime_s"],"first_chunk_compile_and_execution_s":rep.get("first_chunk_compile_and_execution_s",""),"input_load_s":rep.get("input_load_s",""),"correctness_pass":ok,"repeat_rmse_m":rmse,"repeat_minimum_csi":csi,"device":"cuda:0","duration_min":60})
   if not ok: raise SystemExit(f"correctness failed: {backend} {phase} {idx}")
   print(backend,phase,idx,rep["runtime_s"],flush=True)
 with open(OUT/"benchmark_results.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=records[0]); w.writeheader(); w.writerows(records)
 summary={"classification":"secondary_repeated_60min_production_benchmark_not_full_24h_official_benchmark","warmups_per_backend":3,"measurements_per_backend":10,"all_correctness_passed":all(r["correctness_pass"] for r in records),"backends":{}}
 for b in ("cupy","jax"):
  rows=[r for r in records if r["backend"]==b and r["phase"]=="measured"]; summary["backends"][b]=stats([r["solver_runtime_s"] for r in rows]); summary["backends"][b]["median_steps_per_s"]=float(np.median([r["steps_per_s"] for r in rows])); summary["backends"][b]["median_simulated_hours_per_wall_hour"]=float(np.median([r["simulated_hours_per_wall_hour"] for r in rows])); summary["backends"][b]["step_counts"]=sorted(set(r["step_count"] for r in rows))
 cm=summary["backends"]["cupy"]["median_s"]; jm=summary["backends"]["jax"]["median_s"]; summary["jax_runtime_difference_percent"]=(jm-cm)/cm*100; summary["cupy_runtime_over_jax_runtime_ratio"]=cm/jm; summary["ratio_direction"]="CuPy median divided by JAX median"
 (OUT/"benchmark_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__": main()
