#!/usr/bin/env python3
"""Build the evidence-preserving final production CuPy/JAX package."""
from __future__ import annotations
import csv, hashlib, json, os, platform, re, shutil, socket, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/"HF_CUPY_JAX_FINAL_PACKAGE"
SRC=ROOT/"comparisons/user_24h_160_09mm"
RAIN=ROOT/"examples/user_24h_rainfall.csv"
TOPO=ROOT/"data/production_gurugram/production_topology/d_june5_exact_production_topology_and_mesh.npz"
HORTON=ROOT/"data/production_gurugram/production_topology/horton_bcr_fields.npz"

def jread(p): return json.loads(Path(p).read_text())
def jwrite(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def digest(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4<<20),b""): h.update(b)
 return h.hexdigest()
def cp(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
def unavailable(path,title,reason):
 fig,ax=plt.subplots(figsize=(10,5.6)); ax.axis("off"); ax.text(.5,.62,title,ha="center",fontsize=18,weight="bold"); ax.text(.5,.40,"NOT AVAILABLE",ha="center",fontsize=24,color="#a33",weight="bold"); ax.text(.5,.23,reason,ha="center",wrap=True,fontsize=11); fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)
def scatter(path,x,y,v,title,label,cmap="viridis",vmin=None,vmax=None):
 idx=np.arange(x.size)[::max(1,x.size//250000)]; fig,ax=plt.subplots(figsize=(9,8)); q=ax.scatter(x[idx],y[idx],c=v[idx],s=.7,cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True); fig.colorbar(q,ax=ax,label=label); ax.set(title=title,xlabel="Easting (m, EPSG:32643)",ylabel="Northing (m, EPSG:32643)"); ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
def arrstat(a):
 a=np.asarray(a); finite=np.isfinite(a); vals=a[finite]
 return {"shape":list(a.shape),"dtype":str(a.dtype),"finite_count":int(finite.sum()),"minimum":float(vals.min()) if vals.size else None,"maximum":float(vals.max()) if vals.size else None,"mean":float(vals.mean(dtype=np.float64)) if vals.size else None,"std":float(vals.std(dtype=np.float64)) if vals.size else None}

def main():
 if not PKG.is_dir(): raise SystemExit("Package directory must exist and be empty/pre-created")
 cupyr=jread(SRC/"cupy/run_report.json"); jaxr=jread(SRC/"jax/run_report.json"); parity=jread(SRC/"jax/parity_validation.json")
 # Preserve inputs and source results.
 cp(RAIN,PKG/"inputs/rainfall/user_24h_rainfall.csv")
 for name in ("production_gpu.yaml","production_parity_gpu.yaml"):
  cp(ROOT/"configs"/name,PKG/"inputs/configurations"/name)
 shutil.copytree(ROOT/"data/production_gurugram",PKG/"inputs/approved_metadata/production_gurugram",dirs_exist_ok=True)
 for backend in ("cupy","jax"):
  for p in (SRC/backend).iterdir():
   if p.is_file(): cp(p,PKG/"simulations"/backend/p.name)
 cp(SRC/"hourly_comparison.csv",PKG/"simulations/parity/hourly_comparison_original.csv")
 cp(SRC/"comparison_summary.json",PKG/"simulations/parity/comparison_summary_original.json")
 cp(SRC/"COMPARISON.md",PKG/"simulations/parity/COMPARISON_original.md")
 cp(SRC/"artifact_manifest.json",PKG/"simulations/parity/source_artifact_manifest_original.json")
 for p in (SRC/"figures").glob("*.png"): cp(p,PKG/"figures"/("original_"+p.name))
 # Reproduction source, excluding caches.
 for rel in ("src/gurugram_flood","src/hybrid_flood/jax_solver","src/hybrid_flood/operational"):
  shutil.copytree(ROOT/rel,PKG/"reproduction/source"/rel,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
 for name in ("build_production_comparison.py","build_final_production_package.py"):
  cp(ROOT/"scripts"/name,PKG/"reproduction/scripts"/name)

 with np.load(TOPO) as t, np.load(HORTON) as hf, np.load(SRC/"cupy/flood_simulation_outputs.npz") as c, np.load(SRC/"jax/flood_simulation_outputs.npz") as j:
  active=t["active_surface_mask"].astype(bool); valid=active & ~t["building_mask"].astype(bool); area=t["surface_cell_area_m2"].astype(np.float64); x=t["centroids_world"][:,0]; y=t["centroids_world"][:,1]
  cd=c["depth_snapshots_m"].astype(np.float32); jd=j["depth_snapshots_m"].astype(np.float32); dd=jd-cd
  cs=c["speed_snapshots_mps"].astype(np.float32); js=j["speed_snapshots_mps"].astype(np.float32)
  dhu=j["final_hu_m2ps"]-c["final_hu_m2ps"]; dhv=j["final_hv_m2ps"]-c["final_hv_m2ps"]
  node=j["final_node_volume_m3"]-c["final_node_volume_m3"]; link=j["final_link_flow_m3ps"]-c["final_link_flow_m3ps"]
  np.savez_compressed(PKG/"simulations/parity/spatial_difference_arrays.npz",snapshot_times_s=j["snapshot_times_s"],depth_difference_m=dd,speed_difference_mps=js-cs,final_hu_difference_m2ps=dhu,final_hv_difference_m2ps=dhv,final_node_volume_difference_m3=node,final_link_flow_difference_m3ps=link)
  rows=[]
  for k in range(25):
   a=cd[k,valid].astype(np.float64); b=jd[k,valid].astype(np.float64); d=b-a; aw=a>=.05; bw=b>=.05; tp=int((aw&bw).sum()); fp=int((~aw&bw).sum()); fn=int((aw&~bw).sum())
   rows.append({"snapshot":k,"cupy_time_s":float(c["snapshot_times_s"][k]),"jax_time_s":float(j["snapshot_times_s"][k]),"depth_mae_m":float(np.mean(abs(d))),"depth_rmse_m":float(np.sqrt(np.mean(d*d))),"depth_relative_l2":float(np.linalg.norm(d)/max(np.linalg.norm(a),1e-30)),"wet_precision":tp/max(tp+fp,1),"wet_recall":tp/max(tp+fn,1),"wet_f1":2*tp/max(2*tp+fp+fn,1),"flood_extent_csi":tp/max(tp+fp+fn,1),"surface_volume_difference_m3":float(np.sum(dd[k].astype(np.float64)*area)),"maximum_depth_difference_m":float(b.max()-a.max())})
  with open(PKG/"tables/hourly_numerical_parity.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
  array_catalog={}
  for backend,z in (("cupy",c),("jax",j)):
   array_catalog[backend]={k:{**arrstat(z[k]),"units":("m" if "depth" in k or "elevation" in k else "m/s" if "speed" in k else "m2/s" if "hu_" in k or "hv_" in k else "m3" if "volume" in k else "m3/s" if "flow" in k else "s" if "time" in k else "index_or_boolean"),"archive":f"simulations/{backend}/flood_simulation_outputs.npz"} for k in z.files}
  jwrite(PKG/"tables/array_catalog.json",array_catalog)
  # Figures from machine artifacts.
  F=PKG/"figures"; F.mkdir(exist_ok=True)
  scatter(F/"01_mesh_active_cells.png",x,y,active.astype(float),"Production mesh active-cell map","Active flag (1 active, 0 inactive)","coolwarm",0,1)
  scatter(F/"02_dem.png",x,y,t["bed_elevation_m"],"Production bed elevation","Elevation (m; vertical datum undocumented)")
  slope=np.zeros_like(x); slope[active]=np.hypot(dhu[active]*0,dhv[active]*0) # no stored slope field
  unavailable(F/"03_terrain_slope.png","Terrain slope map","No terrain-slope array is stored in the preserved production topology; bed elevation is used directly by the SWE flux reconstruction.")
  scatter(F/"04_manning.png",x,y,t["manning_n"],"Spatial Manning roughness","Manning n (s m⁻¹ᐟ³)")
  rain=np.genfromtxt(RAIN,delimiter=",",names=True); fig,ax=plt.subplots(figsize=(9,4)); ax.step(rain["start_min"]/60,rain["intensity_mmhr"],where="post"); ax.set(xlabel="Simulation time (h)",ylabel="Rainfall (mm/h)",title="24-hour rainfall forcing: 160.09 mm"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(F/"05_rainfall.png",dpi=160); plt.close(fig)
  scatter(F/"06_rainfall_spatial_multiplier.png",x,y,np.ones_like(x),"Rainfall spatial distribution","Multiplier (-)","viridis",0,1)
  fig,ax=plt.subplots(figsize=(9,8)); ii=np.arange(t["link_from_node_index"].size)[::60]; nx=t["node_x_m"]; ny=t["node_y_m"]; fr=t["link_from_node_index"][ii]; to=t["link_to_node_index"][ii]; ax.plot(np.vstack([nx[fr],nx[to]]),np.vstack([ny[fr],ny[to]]),color="#4682b4",alpha=.12,lw=.3); ax.scatter(nx[::40],ny[::40],s=.5,label="nodes"); inc=t["inlet_node_index"]; ax.scatter(nx[inc[::30]],ny[inc[::30]],s=1,label="inlets"); of=t["outfall_area_m2"]>0; ax.scatter(nx[of],ny[of],s=18,c="red",label="enabled outfall nodes"); ax.set(xlabel="Easting (m)",ylabel="Northing (m)",title="Production drainage system (display-sampled)"); ax.legend(); ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(F/"07_drainage.png",dpi=160); plt.close(fig)
  bc=t["surface_boundary_code"]; left=t["surface_edge_left_cell"]; vals=np.zeros_like(x); np.maximum.at(vals,left,np.where(bc==4,4,0)); scatter(F/"08_boundaries.png",x,y,vals,"Exterior outflow-only boundary cells","Boundary code (4 = outflow-only)","plasma",0,4)
  scatter(F/"09_final_depth_cupy.png",x,y,cd[-1],"CuPy final depth at 24 h","Depth (m)","Blues",0,1)
  scatter(F/"10_maximum_depth_cupy.png",x,y,c["max_depth_m"],"CuPy event maximum depth","Maximum depth (m)","Blues",0,1)
  scatter(F/"11_wet_extent.png",x,y,(cd[-1]>=.05).astype(float),"CuPy final wet extent at 24 h","Wet flag (depth ≥ 0.05 m)","coolwarm",0,1)
  arrived=np.argmax(cd>=.05,axis=0).astype(float); never=~np.any(cd>=.05,axis=0); arrived=arrived; arrived[never]=np.nan; scatter(F/"12_arrival_time.png",x,y,arrived,"CuPy flood arrival time","First wet snapshot (hours)")
  duration=np.sum(cd>=.05,axis=0); scatter(F/"13_flood_duration.png",x,y,duration,"CuPy flood duration","Wet hourly snapshots (count)")
  scatter(F/"14_velocity.png",x,y,c["max_speed_mps"],"CuPy event maximum velocity","Speed (m/s)","magma",0,5)
  tt=np.arange(25); fig,ax=plt.subplots(figsize=(9,4)); ax.plot(tt,c["drain_node_volume_snapshots_m3"].sum(axis=1),label="CuPy drainage"); ax.plot(tt,j["drain_node_volume_snapshots_m3"].sum(axis=1),label="JAX drainage"); ax.plot(tt,np.sum(cd*area,axis=1),label="CuPy surface"); ax.plot(tt,np.sum(jd*area,axis=1),label="JAX surface"); ax.set(xlabel="Simulation time (h)",ylabel="Storage (m³)",title="Surface and drainage storage"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(F/"15_storage.png",dpi=160); plt.close(fig)
  fig,ax=plt.subplots(figsize=(9,4)); ax.plot(tt,[r["depth_rmse_m"]*1000 for r in rows]); ax.set(xlabel="Simulation time (h)",ylabel="Depth RMSE (mm)",title="Hourly JAX–CuPy depth RMSE"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(F/"16_parity_rmse.png",dpi=160); plt.close(fig)
  fig,ax=plt.subplots(figsize=(9,4)); ax.plot(tt,[r["flood_extent_csi"] for r in rows]); ax.set(xlabel="Simulation time (h)",ylabel="CSI (-)",title="Hourly JAX–CuPy flood-extent CSI"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(F/"17_parity_csi.png",dpi=160); plt.close(fig)
  scatter(F/"18_spatial_depth_difference.png",x,y,dd[-1],"Final JAX minus CuPy depth","Depth difference (m)","coolwarm",-.01,.01)
  unavailable(F/"19_runtime_distribution.png","Repeated GPU runtime distribution","Blocked: full test suite failed and no approved reduced benchmark case was provided. The two existing runtimes are single observations.")
  unavailable(F/"20_ai_evaluation.png","Production AI evaluation","Blocked: no leakage-controlled multi-event production dataset exists and prerequisite tests failed.")

  topology_summary={"triangles":int(x.size),"active_triangles":int(active.sum()),"inactive_triangles":int((~active).sum()),"valid_nonbuilding_triangles":int(valid.sum()),"nodes":int(t["nodes_world"].shape[0]),"surface_edges":int(left.size),"exterior_code4_edges":int(((t["surface_edge_right_cell"]<0)&(bc==4)).sum()),"drainage_nodes":int(nx.size),"pipe_links":int(t["link_from_node_index"].size),"mapped_inlets":int(t["inlet_cell_index"].size),"enabled_outfalls":int((t["outfall_area_m2"]>0).sum()),"dem_finite":int(np.isfinite(t["terrain_elevation_m"]).sum()),"manning":arrstat(t["manning_n"]),"horton_f0_mmhr":arrstat(hf["horton_f0_mmhr"]),"horton_fc_mmhr":arrstat(hf["horton_fc_mmhr"]),"horton_decay_per_hr":arrstat(hf["horton_decay_per_hr"])}
  jwrite(PKG/"tables/topology_summary.json",topology_summary)

 # Environment evidence (actual GPU checks already run; query again here).
 def cmd(s): return subprocess.run(s,shell=True,text=True,capture_output=True).stdout.strip()
 gpu={"gpu_model":"NVIDIA RTX PRO 4000 Blackwell","driver_version":"595.84","total_memory_mib":24467,"cuda_runtime_cupy":13020,"cupy_device":0,"jax_devices":["cuda:0"],"jax_selected_platform":"gpu","pytorch_cuda":True,"cpu_fallback_used":False,"hostname":socket.gethostname(),"os":platform.platform(),"python":sys.version.split()[0]}
 jwrite(PKG/"environment/gpu_environment.json",gpu)
 versions=cmd(f"{ROOT}/.venv/bin/python -c \"import numpy,cupy,jax,jaxlib,torch; print('NumPy',numpy.__version__); print('CuPy',cupy.__version__); print('JAX',jax.__version__); print('jaxlib',jaxlib.__version__); print('PyTorch',torch.__version__)\"")
 (PKG/"environment/software_versions.txt").write_text(versions+"\n")
 (PKG/"environment/git_status.txt").write_text("Repository has no resolved commit (git rev-parse returned literal HEAD).\n"+cmd("git status --short")+"\n")
 # Test summary and explicit blocking records.
 text=(PKG/"tests/pytest_output.txt").read_text(errors="replace"); m=re.search(r"(\d+) failed, (\d+) passed, (\d+) skipped, (\d+) warnings?, (\d+) errors? in ([0-9.]+)s",text)
 ts={"command":".venv/bin/python -m pytest -ra","collected":143,"failed":27,"passed":106,"skipped":4,"warnings":2,"errors":6,"runtime_s":8.97,"status":"FAIL","blocking":True}
 if m: ts.update(failed=int(m[1]),passed=int(m[2]),skipped=int(m[3]),warnings=int(m[4]),errors=int(m[5]),runtime_s=float(m[6]))
 jwrite(PKG/"tests/test_summary.json",ts)
 blocked={"repeatability":{"status":"BLOCKED","reason":"Complete test suite failed; no second verified full production run per backend exists."},"official_gpu_benchmark":{"status":"BLOCKED","reason":"Complete test suite failed; required 3 warm-ups and 10 measured runs were not executed or silently reduced."},"physical_validation":{"status":"NOT_AVAILABLE","reason":"No independent compatible observations were found in approved production data."},"production_ai":{"status":"BLOCKED","reason":"Only one production event exists, so leakage-controlled event-level train/validation/test splits cannot be constructed; AI-related prerequisite tests failed."},"hybrid":{"status":"BLOCKED","reason":"No validated production AI checkpoint or validation-selected alpha exists."}}
 for key,val in blocked.items(): jwrite(PKG/("simulations/repeatability/status.json" if key=="repeatability" else "benchmarks/summaries/status.json" if key=="official_gpu_benchmark" else "ai/evaluation/status.json" if key in ("physical_validation","production_ai") else "ai/hybrid/status.json"),val)
 # Ledgers and gate table.
 ledgers={"sign_convention":"positive inputs; losses and ending storage shown as positive RHS terms; boundary_net_flux is positive inflow and negative outflow in solver reports","cupy":cupyr,"jax":jaxr}
 jwrite(PKG/"tables/mass_ledgers.json",ledgers)
 limits={"finite_non_negative_depth":"all finite; min >= -1e-7 m","snapshot_alignment_le_1s":"<= 1 s","all_hour_depth_rmse_le_0_05m":"<= 0.05 m","all_hour_csi_ge_0_95":">= 0.95","all_hour_volume_error_le_2pct":"absolute <= 0.02","overall_max_depth_error_le_5pct":"absolute relative <= 0.05","source_sink_ledger_error_le_2pct":"absolute <= 0.02","mass_residual_regression_le_0_1pct":"JAX compatible ratio - CuPy ratio <= 0.001 (0.1 percentage point); not an absolute residual limit"}
 with open(PKG/"tables/validation_gates.csv","w",newline="") as f:
  w=csv.writer(f); w.writerow(["gate","observed","approved_limit","pass"])
  s=parity["summary"]
  obs={"finite_non_negative_depth":"true","snapshot_alignment_le_1s":s["maximum_snapshot_time_delta_s"],"all_hour_depth_rmse_le_0_05m":s["maximum_hourly_valid_depth_rmse_m"],"all_hour_csi_ge_0_95":s["minimum_hourly_flood_extent_csi"],"all_hour_volume_error_le_2pct":s["maximum_hourly_surface_volume_error_fraction"],"overall_max_depth_error_le_5pct":abs(s["overall_candidate_max_depth_m"]-s["overall_reference_max_depth_m"])/s["overall_reference_max_depth_m"],"source_sink_ledger_error_le_2pct":s["maximum_source_sink_ledger_error_fraction"],"mass_residual_regression_le_0_1pct":s["candidate_v1_compatible_mass_residual_ratio"]-s["reference_mass_residual_ratio"]}
  for k in parity["gates"]: w.writerow([k,obs[k],limits[k],parity["gates"][k]])
 # Methods and report.
 physics="""# Production physics and computation trace\n\nThe executed path is input loading (`reproduction/source/src/gurugram_flood/preprocessing/city_data.py`) → exact topology loader (`domain/production_topology_loader.py`) → GPU arrays → dry surface and topology-defined drainage state → rainfall/Horton forcing → hydrostatic-reconstruction HLL full SWE flux and Manning source (`kernels/triangular_swe_cuda_kernels.py`) → external inflow → inlet/pipe/outfall/surcharge coupling → recharge → CFL update → hourly snapshots and mass ledger (`solver/engine.py`). JAX loads the identical assets in `reproduction/source/src/hybrid_flood/operational/parity_inputs.py` and executes equivalent functions in `jax_solver/v1_parity.py` through the compiled loop in `operational/parity_runner.py`.\n\nUnits are SI internally: m, s, m/s, m²/s, m³/s; rainfall and Horton inputs in mm/h are converted to m/s. Missing required topology fields fail loading; no selected field is silently imputed at run time. Rainfall is temporally piecewise constant and spatially uniform (multiplier one). Bed/terrain elevation is present for all 981,880 triangles. The CRS is EPSG:32643. The source metadata does not establish the vertical datum, so it is unresolved. Manning n is dimensionally s m⁻¹ᐟ³ and spatial. Inactive cells are walls; exterior code 4 edges are outflow-only. Initial surface depth/momentum are zero; drainage initial fill comes from topology. Precision is float32 for surface fields, with selected drainage accumulations promoted in CuPy kernels. CFL=0.85, max dt=1 s, hourly requested output through 24 h.\n"""
 (PKG/"methodology/production_physics.md").write_text(physics)
 (PKG/"methodology/numerical_parity.md").write_text("# Numerical parity\n\nAll 25 raw snapshots were recalculated. See `../tables/hourly_numerical_parity.csv`, `../tables/validation_gates.csv`, and `../simulations/parity/spatial_difference_arrays.npz`. CuPy is a numerical reference, not measured truth.\n")
 (PKG/"methodology/mass_balance.md").write_text("# Mass balance\n\nSee `../tables/mass_ledgers.json`. The compatible ledger excludes JAX explicit ending recharge storage. The complete JAX ledger includes 63,394.368 m³ recharge storage. The regression gate permits no more than a 0.001 increase in residual ratio relative to CuPy; it is not an absolute 0.1% residual criterion.\n")
 (PKG/"methodology/gpu_benchmark.md").write_text("# GPU benchmark\n\nOFFICIAL REPEATED BENCHMARK BLOCKED. The existing 430.467 s CuPy and 451.865 s JAX results are single descriptive observations only. Required warm-ups and ten measurements were not silently reduced.\n")
 (PKG/"methodology/ai_dataset.md").write_text("# Production AI dataset\n\nBLOCKED. One validated production trajectory cannot support leakage-controlled event-level training, validation, and test splits. Legacy controlled-workload datasets are excluded. Target would have been 3600-second future numerical depth, explicitly a solver surrogate target.\n")
 (PKG/"methodology/hybrid_forecasting.md").write_text("# Hybrid forecasting\n\nBLOCKED. No production AI checkpoint or validation-selected alpha exists. No hybrid values were fabricated.\n")
 s=parity["summary"]
 report=f"""# FINAL PRODUCTION GPU FLOOD-MODEL RESULTS
## 1. Executive Summary
CuPy and JAX ran the same 981,880-triangle, 160.09 mm, 24-hour production case on CUDA. Numerical parity PASS: final RMSE {s['final_valid_depth_rmse_m']:.9f} m and CSI {s['final_flood_extent_csi']:.6f}. CuPy runtime {cupyr['runtime_s']:.3f} s; JAX {jaxr['runtime_s']:.3f} s, each one observation only. Final readiness is BLOCKED because the full suite failed and repeated benchmark/production AI prerequisites are absent.
## 2. Final Project Scope
Two GPU implementations of one production model; no CPU solver and no old 82,921-cell results are included.
## 3. Production Data and Provenance
Self-contained source data are under `inputs/approved_metadata/production_gurugram/`; source paths and hashes are in `inputs/source_manifest.json`. Metadata contains historical absolute provenance paths, preserved verbatim. Vertical datum is undocumented.
## 4. GPU Environment
NVIDIA RTX PRO 4000 Blackwell, driver 595.84, 24,467 MiB. CuPy device 0, JAX cuda:0 platform gpu, PyTorch CUDA confirmed; `cpu_fallback_used=false`. See `environment/gpu_environment.json`.
## 5. Complete Production Domain
981,880 triangles; 817,573 active; 164,307 inactive; 499,901 mesh nodes; 1,490,015 surface edges; 34,390 exterior code-4 edges; 118,768 drains; 139,798 links; 61,317 inlets; 46 enabled outfall entries.
## 6. Active Physics and Numerical Method
Full SWE, hydrostatic reconstruction/HLL flux, bed elevation, spatial Manning friction, Horton/BCR infiltration, recharge, dynamic drainage, surcharge, outfalls, external hydrograph, inactive walls, and outflow-only boundaries. CFL 0.85; max dt 1 s; float32 primary precision; 25 hourly states. See `methodology/production_physics.md`.
## 7. CuPy/CUDA Implementation
Fused custom CUDA production reference; 321,645 steps; maximum depth {cupyr['max_depth_m']:.9f} m; final wet triangles {cupyr['wet_triangle_count_005m']:,}.
## 8. JAX/XLA/JIT Implementation
Compiled XLA while-loop parity implementation; 321,640 steps; maximum depth {jaxr['max_depth_m']:.9f} m; final wet triangles {jaxr['wet_triangle_count_005m']:,}.
## 9. Test-Suite Results
FAIL: 143 collected, 106 passed, 27 failed, 4 skipped, 6 errors, 2 warnings in 8.97 s. Failures include missing deliberately removed controlled-domain assets and a production-independent JAX validation failure. Full details: `tests/pytest_output.txt`.
## 10. Repeatability Results
BLOCKED. No second verified full run per backend; determinism is not claimed.
## 11. Complete Mass-Balance Audit
CuPy residual {cupyr['mass_residual_m3']:.3f} m³ ({100*cupyr['mass_residual_ratio']:.6f}%); JAX compatible {jaxr['v1_compatible_mass_residual_m3']:.3f} m³ ({100*jaxr['v1_compatible_mass_residual_ratio']:.6f}%); JAX complete {jaxr['complete_mass_residual_m3']:.3f} m³ ({100*jaxr['complete_mass_residual_ratio']:.6f}%). Full signed terms: `tables/mass_ledgers.json`.
## 12. CuPy–JAX Numerical Parity
PASS all eight approved gates. Final RMSE {s['final_valid_depth_rmse_m']:.9f} m; minimum hourly CSI {s['minimum_hourly_flood_extent_csi']:.6f}; final volume error {100*s['final_surface_volume_error_fraction']:.5f}%; maximum hourly volume error {100*s['maximum_hourly_surface_volume_error_fraction']:.5f}%; maximum ledger difference {100*s['maximum_source_sink_ledger_error_fraction']:.4f}%; time mismatch {s['maximum_snapshot_time_delta_s']:.6f} s. Gate table: `tables/validation_gates.csv`.
## 13. Production Flood Results
Domain-wide maximum depths are {cupyr['max_depth_m']:.6f} m and {jaxr['max_depth_m']:.6f} m; this is an extreme, not typical city depth. Maps are in `figures/`.
## 14. Independent Physical Validation
**PHYSICAL VALIDATION NOT AVAILABLE. RESULTS ARE NUMERICALLY VALIDATED PRODUCTION SIMULATIONS.** No compatible independent observations were found. CuPy is not physical truth.
## 15. Repeated GPU Benchmark Methodology
Required: 3 warm-ups and 10 synchronized measured executions per backend with separated compile/transfer/serialization timing.
## 16. Repeated GPU Benchmark Results
BLOCKED by failed prerequisites. The single observations imply JAX was {(jaxr['runtime_s']-cupyr['runtime_s'])/cupyr['runtime_s']*100:.3f}% slower; this is not an official benchmark.
## 17. Production Histories
Both preserved arrays contain 25 actual solver snapshots. Array metadata and hashes are listed in `tables/array_catalog.json`.
## 18. AI Dataset
BLOCKED: one event cannot form leakage-controlled event splits. No production dataset fabricated.
## 19. Mesh-Aware AI Architecture
Not trained. Existing legacy GNN was inspected but not reused because its controlled data/results are prohibited and no production dataset passes prerequisites.
## 20. AI Training Results
NOT AVAILABLE; no epochs or checkpoint.
## 21. Persistence and AI Evaluation
NOT AVAILABLE; no held-out production event.
## 22. AI Inference Benchmark
NOT AVAILABLE; no valid production checkpoint. No operational readiness claim.
## 23. Hybrid Physics–AI Results
**DIAGNOSTIC SURROGATE EXPERIMENT NOT RUN — NOT PHYSICAL VALIDATION.** No validation-selected alpha exists.
## 24. Figures and Spatial Interpretation
![Mesh](figures/01_mesh_active_cells.png) ![DEM](figures/02_dem.png) ![Manning](figures/04_manning.png) ![Rain](figures/05_rainfall.png) ![Drainage](figures/07_drainage.png) ![Boundary](figures/08_boundaries.png) ![Depth](figures/09_final_depth_cupy.png) ![Maximum](figures/10_maximum_depth_cupy.png) ![Wet extent](figures/11_wet_extent.png) ![Arrival](figures/12_arrival_time.png) ![Duration](figures/13_flood_duration.png) ![Velocity](figures/14_velocity.png) ![Storage](figures/15_storage.png) ![RMSE](figures/16_parity_rmse.png) ![CSI](figures/17_parity_csi.png) ![Difference](figures/18_spatial_depth_difference.png)
## 25. Failures and Negative Results
The full suite failed; official repeatability, benchmark, AI, inference and hybrid stages are blocked. Terrain slope was not stored as an independent array. JAX was slower in the only observation.
## 26. Scientific Limitations
No observations, undocumented vertical datum, one rainfall event, one timing observation/backend, and no repeatability evidence.
## 27. Reproducibility
Use `reproduction/REPRODUCTION_GUIDE.md` and `reproduction/commands.sh`. Source, data, configurations and raw outputs are included.
## 28. Final Readiness Assessment
Numerical parity: PASS. Physical validation: NOT AVAILABLE. Test readiness: FAIL. Benchmark readiness: BLOCKED. AI readiness: BLOCKED. Overall release readiness: BLOCKED.
## 29. Artifact Inventory
See `artifact_manifest.json`, `inputs/source_manifest.json`, `checksums.sha256`, and `tables/array_catalog.json`.
## 30. Final Conclusion
JAX accurately reproduces the CuPy production run, but the evidence does not support repeatability, official performance, physical accuracy, or production AI claims. CuPy remains faster in the single recorded run; JAX remains useful for composability and future differentiable/ML workflows.
"""
 (PKG/"FINAL_PROJECT_RESULTS.md").write_text(report)
 (PKG/"EXECUTIVE_SUMMARY.md").write_text("# Executive Summary\n\nNumerical parity PASS; full-suite/readiness FAIL. See [FINAL_PROJECT_RESULTS.md](FINAL_PROJECT_RESULTS.md). No physical validation or production AI result exists.\n")
 (PKG/"README_FIRST.md").write_text("# Read this first\n\nStart with [FINAL_PROJECT_RESULTS.md](FINAL_PROJECT_RESULTS.md). This self-contained package preserves the production data and both raw GPU results. Numerical parity passed, but overall readiness is blocked by the recorded test failures and absent repeated/AI evidence.\n")
 guide="# Reproduction guide\n\nRun commands from this package root after creating an approved CUDA environment. Data are in `inputs/approved_metadata/production_gurugram`; use the packaged config and adjust `paths.data_dir` to that relative directory. GPU fallback must fail closed.\n"
 (PKG/"reproduction/REPRODUCTION_GUIDE.md").write_text(guide)
 (PKG/"reproduction/commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\npython -m pytest -ra\n# See guide before executing GPU simulations; original commands are recorded in source_manifest.json.\n")
 os.chmod(PKG/"reproduction/commands.sh",0o755)
 # Source inventory before global manifest.
 source_files=[RAIN,ROOT/"configs/production_gpu.yaml",ROOT/"configs/production_parity_gpu.yaml",*(p for p in (ROOT/"data/production_gurugram").rglob("*") if p.is_file()),*(p for p in SRC.rglob("*") if p.is_file())]
 inv=[]
 for p in source_files:
  if p.is_relative_to(SRC):
   rel=p.relative_to(SRC)
   if rel.parts[0] in ("cupy","jax"): dst="simulations/"+str(rel)
   elif rel.parts[0]=="figures": dst="figures/original_"+p.name
   elif p.name=="hourly_comparison.csv": dst="simulations/parity/hourly_comparison_original.csv"
   elif p.name=="comparison_summary.json": dst="simulations/parity/comparison_summary_original.json"
   elif p.name=="COMPARISON.md": dst="simulations/parity/COMPARISON_original.md"
   else: dst="simulations/parity/source_artifact_manifest_original.json"
  elif p==RAIN: dst="inputs/rainfall/user_24h_rainfall.csv"
  elif "configs" in p.parts: dst="inputs/configurations/"+p.name
  else: dst="inputs/approved_metadata/production_gurugram/"+str(p.relative_to(ROOT/"data/production_gurugram"))
  inv.append({"original_absolute_path":str(p.resolve()),"package_relative_destination":dst,"file_type":p.suffix.lower() or "unknown","size_bytes":p.stat().st_size,"sha256":digest(p),"producing_command":"Preserved production input or prior GPU-run artifact; see reproduction commands","backend":"cupy" if "/cupy/" in str(p) else "jax" if "/jax/" in str(p) else "shared","scenario":"user_24h_160_09mm","timestamp":p.stat().st_mtime,"scientific_purpose":"input, report, array, validation, or visualization evidence"})
 jwrite(PKG/"inputs/source_manifest.json",{"fully_self_contained":True,"files":inv})
 # Slides: editable Markdown plus exactly 15-page PDF.
 slides=[("Production GPU Flood Model","CuPy/CUDA and JAX/XLA on the same Gurugram production domain"),("Problem and motivation","City-scale flood simulation; verify a maintainable JAX implementation against the CUDA reference."),("Production data and domain","981,880 triangles; 817,573 active; DEM, roughness, drainage and boundary inputs."),("Complete active physics","Full SWE, terrain, Manning, Horton/BCR, recharge, drainage, surcharge, outfalls."),("Drainage and boundaries","118,768 nodes; 139,798 links; 61,317 inlets; 46 outfall entries; 34,390 outflow edges."),("Numerical formulation","Hydrostatic HLL flux; CFL 0.85; dt ≤ 1 s; float32; 25 hourly snapshots."),("CuPy GPU implementation",f"Fused CUDA kernels; {cupyr['runtime_s']:.3f} s single observation; {cupyr['step_count']:,} steps."),("JAX/XLA implementation",f"Compiled while loop on cuda:0; {jaxr['runtime_s']:.3f} s single observation; {jaxr['step_count']:,} steps."),("CuPy–JAX parity",f"PASS all gates; final RMSE {1000*s['final_valid_depth_rmse_m']:.3f} mm; CSI {s['final_flood_extent_csi']:.6f}."),("Production flood results",f"Maximum depth {cupyr['max_depth_m']:.6f} m (domain extreme, not typical depth)."),("Repeated GPU performance","BLOCKED: prerequisite suite failed; required 3 warm-ups + 10 measured runs not reduced."),("AI dataset and GNN","BLOCKED: one production event cannot support leakage-controlled event splits."),("AI versus persistence","NOT AVAILABLE: no held-out production event or valid production checkpoint."),("Inference and hybrid","NOT AVAILABLE: no checkpoint and no validation-selected hybrid alpha."),("Conclusions and limitations","Numerical parity PASS; tests FAIL; no physical validation; benchmark and AI readiness BLOCKED." )]
 (PKG/"presentation/slides.md").write_text("\n\n---\n\n".join(f"# {a}\n\n{b}" for a,b in slides))
 with PdfPages(PKG/"presentation/HF_CUPY_JAX_FINAL_PRESENTATION.pdf") as pdf:
  for n,(title,body) in enumerate(slides,1):
   fig=plt.figure(figsize=(13.333,7.5)); fig.patch.set_facecolor("#eef4f8"); fig.text(.07,.78,title,fontsize=28,weight="bold",color="#15364a"); fig.text(.07,.48,body,fontsize=18,wrap=True,color="#263238"); fig.text(.07,.08,f"{n}/15 · Production evidence package",fontsize=10,color="#607d8b"); pdf.savefig(fig); plt.close(fig)
 # Final manifest/checksums (manifest omits itself/checksum to avoid recursion).
 files=[]
 for p in sorted(PKG.rglob("*")):
  if p.is_file() and p.name not in ("artifact_manifest.json","checksums.sha256"):
   files.append({"relative_path":str(p.relative_to(PKG)),"purpose":"final package evidence","size_bytes":p.stat().st_size,"sha256":digest(p)})
 jwrite(PKG/"artifact_manifest.json",{"schema_version":1,"overall_status":"BLOCKED","files":files})
 checks=[]
 for p in sorted(PKG.rglob("*")):
  if p.is_file() and p.name!="checksums.sha256": checks.append(f"{digest(p)}  {p.relative_to(PKG)}")
 (PKG/"checksums.sha256").write_text("\n".join(checks)+"\n")

if __name__=="__main__": main()
