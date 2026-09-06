#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"completed_work/repeatability"

def sha(p):
 h=hashlib.sha256();
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4<<20),b""): h.update(b)
 return h.hexdigest()
def report(p): return json.loads((p/"run_report.json").read_text())
def compare(a,b):
 with np.load(a/"flood_simulation_outputs.npz") as x,np.load(b/"flood_simulation_outputs.npz") as y:
  d=y["depth_snapshots_m"].astype(np.float64)-x["depth_snapshots_m"].astype(np.float64); valid=y["active_surface_mask"].astype(bool)&~y["building_mask"].astype(bool); dv=d[:,valid]; xa=x["depth_snapshots_m"][:,valid].astype(np.float64); aw=xa>=.05; bw=y["depth_snapshots_m"][:,valid]>=.05; inter=aw&bw; union=aw|bw
  mom={k:{"maximum_absolute":float(np.max(np.abs(y[k].astype(np.float64)-x[k].astype(np.float64)))),"rmse":float(np.sqrt(np.mean((y[k].astype(np.float64)-x[k].astype(np.float64))**2)))} for k in ("final_hu_m2ps","final_hv_m2ps")}
  area=np.load(ROOT/"data/production_gurugram/production_topology/d_june5_exact_production_topology_and_mesh.npz")["surface_cell_area_m2"].astype(np.float64)
  return {"byte_level_identical":sha(a/"flood_simulation_outputs.npz")==sha(b/"flood_simulation_outputs.npz"),"maximum_absolute_depth_difference_m":float(np.max(np.abs(dv))),"mean_absolute_depth_difference_m":float(np.mean(np.abs(dv))),"depth_rmse_m":float(np.sqrt(np.mean(dv*dv))),"relative_l2_depth":float(np.linalg.norm(dv)/max(np.linalg.norm(xa),1e-30)),"momentum":mom,"minimum_snapshot_flood_extent_csi":float(np.min(inter.sum(axis=1)/np.maximum(union.sum(axis=1),1))),"all_snapshot_wet_cell_exact_agreement":bool(np.array_equal(aw,bw)),"maximum_depth_difference_m":float(y["max_depth_m"].max()-x["max_depth_m"].max()),"final_surface_volume_difference_m3":float(np.sum((y["final_depth_m"]-x["final_depth_m"]).astype(np.float64)*area)),"final_drainage_storage_difference_m3":float(y["final_node_volume_m3"].astype(np.float64).sum()-x["final_node_volume_m3"].astype(np.float64).sum()),"maximum_snapshot_time_difference_s":float(np.max(np.abs(y["snapshot_times_s"]-x["snapshot_times_s"])))}

def main():
 pairs={"cupy":(ROOT/"comparisons/user_24h_160_09mm/cupy",OUT/"cupy_B"),"jax":(ROOT/"comparisons/user_24h_160_09mm/jax",OUT/"jax_B")}; result={"existing_approved_repeatability_criteria":{"depth_rmse_m_max":.05,"minimum_csi":.95,"snapshot_alignment_s_max":1},"backends":{}}
 for name,(a,b) in pairs.items():
  ra,rb=report(a),report(b); m=compare(a,b); ledger={k:float(rb.get(k,0)-ra.get(k,0)) for k in set(ra)&set(rb) if k.endswith("_m3") and isinstance(ra[k],(int,float)) and isinstance(rb[k],(int,float))}; m.update({"run_A":{"path":str(a),"runtime_s":ra["runtime_s"],"steps":ra["step_count"],"npz_sha256":sha(a/"flood_simulation_outputs.npz")},"run_B":{"path":str(b),"runtime_s":rb["runtime_s"],"steps":rb["step_count"],"npz_sha256":sha(b/"flood_simulation_outputs.npz")},"timestep_count_difference":rb["step_count"]-ra["step_count"],"mass_residual_ratio_difference":float(rb.get("mass_residual_ratio",rb.get("v1_compatible_mass_residual_ratio"))-ra.get("mass_residual_ratio",ra.get("v1_compatible_mass_residual_ratio"))),"ledger_differences_m3":ledger}); m["classification"]="bitwise_deterministic" if m["byte_level_identical"] else "numerically_deterministic_within_existing_parity_criteria" if m["depth_rmse_m"]<=.05 and m["minimum_snapshot_flood_extent_csi"]>=.95 and m["maximum_snapshot_time_difference_s"]<=1 else "not_deterministic"; result["backends"][name]=m
 (OUT/"repeatability_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
 md="# Full-production repeatability report\n\nTwo independent 24-hour executions were compared for each backend. No new tolerance was invented; classification uses the existing parity criteria (RMSE ≤0.05 m, CSI ≥0.95, snapshot offset ≤1 s). Byte-level identity is reported separately.\n\n"
 for n,m in result["backends"].items(): md+=f"## {n.upper()}\n\nClassification: **{m['classification']}**. Byte-identical archive: {m['byte_level_identical']}. RMSE {m['depth_rmse_m']:.9g} m; max absolute depth difference {m['maximum_absolute_depth_difference_m']:.9g} m; minimum CSI {m['minimum_snapshot_flood_extent_csi']:.9g}; timestep difference {m['timestep_count_difference']:+d}; maximum snapshot offset {m['maximum_snapshot_time_difference_s']:.6g} s.\n\n"
 md+="Differences correlate with timestep-count/snapshot-time divergence. The available evidence does not isolate the lower-level cause among GPU reduction ordering and wet/dry transition sensitivity, so no stronger causal claim is made.\n"
 (OUT/"repeatability_report.md").write_text(md)
if __name__=="__main__": main()
