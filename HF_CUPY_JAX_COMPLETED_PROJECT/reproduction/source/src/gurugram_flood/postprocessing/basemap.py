"""Standalone interactive basemap HTML with an hour slider (Bokeh).

Optional artifact: a self-contained HTML file showing hourly flood depth
polygons over a CartoDB basemap. Useful for QA and stakeholder handoff;
the frontend team normally consumes the GeoJSONs instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..logging_utils import get_logger
from ..solver.engine import SimulationResult
from .geojson_export import select_hourly_snapshots

log = get_logger("basemap")

BASEMAP_HTML_NAME = "hourly_depth_basemap.html"


def export_basemap_html(result: SimulationResult, out_dir: Path) -> Path | None:
    """Write the hourly-slider basemap HTML. Returns None if Bokeh is missing."""
    try:
        from bokeh.embed import file_html
        from bokeh.layouts import column
        from bokeh.models import (
            BasicTicker,
            ColorBar,
            ColumnDataSource,
            CustomJS,
            Div,
            HoverTool,
            LinearColorMapper,
            Slider,
            WMTSTileSource,
        )
        from bokeh.palettes import Blues9
        from bokeh.plotting import figure
        from bokeh.resources import CDN
        from pyproj import Transformer
    except ImportError:
        log.warning("bokeh not installed; skipping basemap HTML")
        return None

    cfg = result.config
    mesh = result.city.mesh
    threshold_m = cfg.outputs.geojson_threshold_m
    color_max_m = cfg.outputs.basemap_color_max_m
    max_features = cfg.outputs.basemap_max_features

    valid = mesh.active_mask & ~mesh.building_mask
    selection = select_hourly_snapshots(result.snapshot_times_min)
    target_hours = selection.target_hours
    hourly_depth_m = result.depth_snapshots_m[selection.snapshot_indices]
    peak_depth = np.nanmax(hourly_depth_m, axis=0)

    wet_any = valid & np.isfinite(peak_depth) & (peak_depth >= threshold_m)
    wet_idx = np.where(wet_any)[0]
    if len(wet_idx) > max_features:
        # Always keep deep cells; subsample the shallow ones deterministically.
        deep_idx = wet_idx[peak_depth[wet_idx] >= 1.0]
        shallow_idx = wet_idx[peak_depth[wet_idx] < 1.0]
        n_keep = max(0, max_features - len(deep_idx))
        rng = np.random.default_rng(42)
        shallow_keep = rng.choice(shallow_idx, size=min(n_keep, len(shallow_idx)), replace=False)
        plot_idx = np.sort(np.concatenate([deep_idx, shallow_keep]))
    else:
        plot_idx = wet_idx
    log.info("Basemap: plotting %s of %s wet triangles", f"{len(plot_idx):,}", f"{len(wet_idx):,}")

    wm_tx = Transformer.from_crs(cfg.outputs.mesh_crs, "EPSG:3857", always_xy=True)
    nodes_world = mesh.nodes_xy.astype(np.float64)
    mx_nodes, my_nodes = wm_tx.transform(nodes_world[:, 0], nodes_world[:, 1])
    tri_verts = mesh.triangles[plot_idx]
    xs = [mx_nodes[v].tolist() for v in tri_verts]
    ys = [my_nodes[v].tolist() for v in tri_verts]

    timestamps = result.rainfall.timestamps_by_hour
    intensities = result.rainfall.intensity_by_hour
    wet_counts = [
        int((valid & np.isfinite(d) & (d >= threshold_m)).sum()) for d in hourly_depth_m
    ]

    src_dict = {
        "xs": xs,
        "ys": ys,
        "cell_id": plot_idx.astype(int).tolist(),
        "peak_depth": peak_depth[plot_idx].astype(float).tolist(),
        "depth": hourly_depth_m[0, plot_idx].clip(0, color_max_m).astype(float).tolist(),
        "depth_raw": hourly_depth_m[0, plot_idx].astype(float).tolist(),
    }
    for row_i, hour in enumerate(target_hours):
        src_dict[f"depth_h{int(hour):02d}"] = hourly_depth_m[row_i, plot_idx].clip(0, color_max_m).astype(float).tolist()
        src_dict[f"depth_raw_h{int(hour):02d}"] = hourly_depth_m[row_i, plot_idx].astype(float).tolist()
    hourly_source = ColumnDataSource(src_dict)

    ts0 = timestamps.get(int(target_hours[0]), "")
    inm0 = intensities.get(int(target_hours[0]), 0.0)

    p = figure(
        title=f"Flood depth  |  Hour 0  |  {ts0}  |  {inm0:.1f} mm/hr",
        x_range=(float(mx_nodes.min() - 1500), float(mx_nodes.max() + 1500)),
        y_range=(float(my_nodes.min() - 1500), float(my_nodes.max() + 1500)),
        x_axis_type="mercator",
        y_axis_type="mercator",
        width=1050,
        height=880,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p.add_tile(WMTSTileSource(
        url="https://a.basemaps.cartocdn.com/light_all/{Z}/{X}/{Y}.png",
        attribution="(c) CartoDB (c) OpenStreetMap contributors",
    ))
    mapper = LinearColorMapper(
        palette=list(reversed(Blues9))[1:], low=0.0, high=color_max_m,
        nan_color="rgba(0,0,0,0)",
    )
    patches = p.patches(
        xs="xs", ys="ys", source=hourly_source,
        fill_color={"field": "depth", "transform": mapper},
        fill_alpha=0.88, line_color=None,
    )
    p.add_tools(HoverTool(renderers=[patches], tooltips=[
        ("Cell", "@cell_id"),
        ("Depth (m)", "@depth_raw{0.000}"),
        ("Peak (m)", "@peak_depth{0.000}"),
    ]))
    p.add_layout(
        ColorBar(color_mapper=mapper, ticker=BasicTicker(desired_num_ticks=8),
                 title="Depth (m)", width=14, height=240),
        "right",
    )
    p.grid.visible = False

    slider = Slider(start=0, end=len(target_hours) - 1, value=0, step=1, title="Hour")
    info = Div(
        text=(f"<b>IST:</b> {ts0} &nbsp;&nbsp; <b>Intensity:</b> {inm0:.1f} mm/hr "
              f"&nbsp;&nbsp; <b>Wet cells:</b> {wet_counts[0]:,}"),
        styles={"font-size": "13px", "padding": "4px 0"},
    )
    callback = CustomJS(
        args=dict(
            source=hourly_source, slider=slider, plot=p, info=info,
            hour_labels=[int(h) for h in target_hours],
            timestamps=[timestamps.get(int(h), "") for h in target_hours],
            intensities=[float(intensities.get(int(h), 0.0)) for h in target_hours],
            wet_counts=wet_counts,
        ),
        code="""
            const i = slider.value;
            const h = String(hour_labels[i]).padStart(2, '0');
            source.data['depth']     = source.data['depth_h' + h];
            source.data['depth_raw'] = source.data['depth_raw_h' + h];
            source.change.emit();
            plot.title.text = 'Flood depth  |  Hour ' + hour_labels[i]
                            + '  |  ' + timestamps[i]
                            + '  |  ' + intensities[i].toFixed(1) + ' mm/hr';
            info.text = '<b>IST:</b> ' + timestamps[i]
                      + '&nbsp;&nbsp; <b>Intensity:</b> ' + intensities[i].toFixed(1) + ' mm/hr'
                      + '&nbsp;&nbsp; <b>Wet cells:</b> ' + wet_counts[i].toLocaleString();
        """,
    )
    slider.js_on_change("value", callback)

    layout = column(slider, info, p)
    html_path = out_dir / BASEMAP_HTML_NAME
    html_path.write_text(file_html(layout, CDN, "Gurugram Hourly Flood Depth"), encoding="utf-8")
    log.info("Saved basemap HTML: %s", html_path)
    return html_path
