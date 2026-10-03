"""Render stored demo arrays as scientific figures and an offline browser demo.

The vector fields are evaluations of the frozen *marginal* models. The noise
panels use inverse marginal-flow coordinates, not the joint generator's source.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


METHOD_LABELS = {
    "empirical_optimal": "Empirical optimum",
    "antithetic": "Antithetic",
    "independent": "Independent",
    "learned": "Learned joint flow",
    "conventional": "Conventional transport",
    "conventional_rf": "Conventional transport",
    "ordinary_transport": "Conventional 2D transport",
}


def _label(name: str) -> str:
    return METHOD_LABELS.get(name, name.replace("_", " ").capitalize())


def _load_display(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {
        "times", "field_points", "field_x", "field_y", "bounds",
        "method_names", "output_x", "output_y", "marginal_path_x",
        "marginal_path_y", "noise_x", "noise_y", "component_pair_matrices",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"display.npz is missing {', '.join(sorted(missing))}")
    times = data["times"]
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Display times must be a strictly increasing vector")
    if not np.isclose(times[0], 0) or not np.isclose(times[-1], 1):
        raise ValueError("Display trajectories must run from time 0 to time 1")
    names = data["method_names"]
    if names.ndim != 1 or not len(names) or names.dtype.kind not in "US":
        raise ValueError("method_names must be a nonempty string vector")
    if len(set(names.tolist())) != len(names):
        raise ValueError("Display method names must be unique")
    count = len(names)
    output = data["output_x"]
    if output.ndim != 3 or output.shape[0] != count or output.shape[2] != 2 or not output.shape[1]:
        raise ValueError("output_x must have shape (methods, samples, 2)")
    samples = output.shape[1]
    for key in ("output_x", "output_y", "noise_x", "noise_y"):
        if data[key].shape != (count, samples, 2):
            raise ValueError(f"{key} has an incompatible shape")
    for key in ("marginal_path_x", "marginal_path_y"):
        if data[key].shape != (count, len(times), samples, 2):
            raise ValueError(f"{key} must have shape (methods, times, samples, 2)")
    points = data["field_points"]
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("field_points must have shape (grid points, 2)")
    for key in ("field_x", "field_y"):
        if data[key].shape != (len(times), len(points), 2):
            raise ValueError(f"{key} must have shape (times, grid points, 2)")
    if data["bounds"].shape != (4,) or np.any(data["bounds"][[1, 3]] <= data["bounds"][[0, 2]]):
        raise ValueError("bounds must be [xmin, xmax, ymin, ymax] with positive ranges")
    matrices = data["component_pair_matrices"]
    if matrices.ndim != 3 or matrices.shape[0] != count or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("component_pair_matrices must contain a square matrix per method")
    for key in required.difference({"method_names"}):
        if data[key].dtype.kind not in "fiu" or not np.isfinite(data[key]).all():
            raise ValueError(f"{key} must contain finite numeric values")
    optional_paths = ("ordinary_linear_path", "ordinary_path")
    if any(key in data for key in optional_paths):
        if not all(key in data for key in optional_paths):
            raise ValueError("ordinary_linear_path and ordinary_path must be supplied together")
        linear = data["ordinary_linear_path"]
        if linear.ndim != 3 or linear.shape[0] != len(times) or not linear.shape[1] or linear.shape[2] != 2:
            raise ValueError("ordinary_linear_path must have shape (times, samples, 2)")
        if data["ordinary_path"].shape != linear.shape:
            raise ValueError("ordinary_path must match ordinary_linear_path's shape")
        for key in optional_paths:
            if data[key].dtype.kind not in "fiu" or not np.isfinite(data[key]).all():
                raise ValueError(f"{key} must contain finite numeric values")
        if not np.allclose(linear[0], data["ordinary_path"][0], rtol=1e-5, atol=1e-6):
            raise ValueError("Ordinary interpolation and ODE paths must have the same starting points")
    for key in ("reference_x", "reference_y"):
        if key in data:
            values = data[key]
            if values.ndim != 2 or values.shape[1] != 2 or not len(values):
                raise ValueError(f"{key} must have shape (samples, 2)")
            if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                raise ValueError(f"{key} must contain finite numeric values")
    return data


def _point_bounds(*arrays: np.ndarray) -> np.ndarray:
    points = np.concatenate([array.reshape(-1, 2) for array in arrays])
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float(np.max(hi - lo)) * 0.56, 0.5)
    return np.array([center[0] - radius, center[0] + radius, center[1] - radius, center[1] + radius])


def _reward(report: dict[str, Any], name: str, matrix: np.ndarray) -> float:
    metrics = report.get("methods", {}).get(name, {})
    for key in ("mismatch_reward", "component_mismatch", "different_component_probability"):
        if key in metrics:
            return float(metrics[key])
    return float(matrix.sum() - np.trace(matrix))


def _figures(output_dir: Path, data: dict[str, np.ndarray], report: dict[str, Any], title: str) -> list[Path]:
    # Import lazily so numerical/headless runners do not initialize pyplot.
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="diverse-coupling-mpl-")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    names = data["method_names"].tolist()
    method = names.index("learned") if "learned" in names else 0
    sample_count = data["output_x"].shape[1]
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, sample_count))
    step = int(np.argmin(np.abs(data["times"] - 0.5)))
    time = float(data["times"][step])

    def configure(ax, bounds, subtitle):
        ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect="equal", title=subtitle, xlabel="Coordinate 1", ylabel="Coordinate 2")
        ax.grid(alpha=0.13)

    def pairs(ax, x, y):
        ax.add_collection(LineCollection(np.stack((x, y), axis=1), colors=colors, linewidths=0.8, alpha=0.45, linestyles="dashed"))
        ax.scatter(x[:, 0], x[:, 1], c=colors, s=24, marker="o", label="X", edgecolors="none", zorder=3)
        ax.scatter(y[:, 0], y[:, 1], edgecolors=colors, facecolors="none", s=36, marker="^", linewidths=1.2, label="Y", zorder=3)

    def reference(ax):
        for key in ("reference_x", "reference_y"):
            if key in data:
                points = data[key]
                ax.scatter(points[:, 0], points[:, 1], color="#8995a6", s=5, alpha=0.13, edgecolors="none", zorder=1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.2), constrained_layout=True)
    fig.suptitle(f"{title} — {_label(names[method])}", fontsize=15)
    configure(axes[0, 0], data["bounds"], "Coupled outputs (dashed lines identify pairs)")
    reference(axes[0, 0])
    pairs(axes[0, 0], data["output_x"][method], data["output_y"][method])
    axes[0, 0].legend(loc="upper right", fontsize=9)
    noise_bounds = _point_bounds(data["noise_x"], data["noise_y"])
    configure(axes[0, 1], noise_bounds, "Marginal noise: inverse of frozen 2D flows")
    pairs(axes[0, 1], data["noise_x"][method], data["noise_y"][method])
    for ax, side in zip(axes[1], ("x", "y")):
        configure(ax, data["bounds"], f"Frozen {side.upper()} marginal velocity, t = {time:.2f}")
        points, field = data["field_points"], data[f"field_{side}"][step]
        norms = np.linalg.norm(data[f"field_{side}"], axis=-1)
        span = max(np.diff(data["bounds"].reshape(2, 2), axis=1).ravel())
        scale = max(float(np.quantile(norms, 0.95)) / (0.075 * span), 1e-8)
        ax.quiver(points[:, 0], points[:, 1], field[:, 0], field[:, 1], color="#40516b", alpha=0.65, angles="xy", scale_units="xy", scale=scale, width=0.003)
        paths = data[f"marginal_path_{side}"][method]
        for index in range(min(sample_count, 24)):
            ax.plot(paths[:step + 1, index, 0], paths[:step + 1, index, 1], color=colors[index], alpha=0.7, linewidth=1.0)
        ax.scatter(paths[step, :, 0], paths[step, :, 1], c=colors, s=14, marker="o" if side == "x" else "^", zorder=3)
    overview = output_dir / "overview.png"
    fig.savefig(overview, dpi=180)
    plt.close(fig)

    count = len(names)
    fig = plt.figure(figsize=(max(10, count * 3), 7), constrained_layout=True)
    grid = fig.add_gridspec(2, count, height_ratios=[1.1, 1])
    fig.suptitle(f"{title} — coupling diagnostics", fontsize=15)
    matrices = data["component_pair_matrices"]
    maximum = max(float(matrices.max()), 1e-10)
    for index, name in enumerate(names):
        ax = fig.add_subplot(grid[0, index])
        image = ax.imshow(matrices[index], origin="upper", vmin=0, vmax=maximum, cmap="Blues")
        components = matrices.shape[-1]
        ax.set(title=_label(name), xlabel="Y component", ylabel="X component", xticks=np.arange(components), yticks=np.arange(components))
        if components <= 8:
            for row in range(components):
                for column in range(components):
                    value = matrices[index, row, column]
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > maximum * 0.55 else "#17233a")
        fig.colorbar(image, ax=ax, shrink=0.7, label="Joint probability")
    ax = fig.add_subplot(grid[1, :])
    rewards = [_reward(report, name, matrices[index]) for index, name in enumerate(names)]
    ax.bar(np.arange(count), rewards, color=["#3b6ea8", "#e18839", "#8595a6", "#329b83"][:count] if count <= 4 else "#3b6ea8")
    ax.set(xticks=np.arange(count), xticklabels=[_label(name) for name in names], ylim=(0, max(1.05, max(rewards) * 1.1)), ylabel="Different-component probability")
    ax.grid(axis="y", alpha=0.2)
    for index, reward in enumerate(rewards):
        ax.text(index, reward + 0.025, f"{reward:.3f}", ha="center")
    diagnostics = output_dir / "diagnostics.png"
    fig.savefig(diagnostics, dpi=180)
    plt.close(fig)
    figures = [overview, diagnostics]
    if "ordinary_path" in data:
        linear, ode = data["ordinary_linear_path"], data["ordinary_path"]
        ordinary_colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, linear.shape[1]))
        bounds = _point_bounds(linear, ode)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), constrained_layout=True)
        fig.suptitle(f"{title} — conventional 2D transport", fontsize=15)
        for ax, paths, subtitle, endpoint_label in zip(
            axes, (linear, ode),
            ("Straight interpolation of training pairs", "Ordinary 2D RF: induced ODE trajectories"),
            ("Paired target Y", "ODE endpoint"),
        ):
            configure(ax, bounds, subtitle)
            reference(ax)
            for index in range(min(paths.shape[1], 48)):
                ax.plot(paths[:, index, 0], paths[:, index, 1], color=ordinary_colors[index], alpha=0.65, linewidth=1.2)
            ax.scatter(paths[0, :, 0], paths[0, :, 1], c=ordinary_colors, s=20, marker="o", edgecolors="none", label="Source X", zorder=3)
            ax.scatter(paths[-1, :, 0], paths[-1, :, 1], edgecolors=ordinary_colors, facecolors="none", s=28, marker="^", linewidths=1.1, label=endpoint_label, zorder=3)
            ax.legend(fontsize=9, loc="upper right")
        fig.supxlabel("Color identifies the same source observation. Transport begins in left data space; these are not joint 4D projections.", fontsize=10)
        ordinary = output_dir / "ordinary_transport.png"
        fig.savefig(ordinary, dpi=180)
        plt.close(fig)
        figures.append(ordinary)
    return figures


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#17233a;background:#f2f5f8}*{box-sizing:border-box}body{margin:0;padding:24px}main{max-width:1220px;margin:auto}h1{font-size:25px;margin:0 0 8px}p{line-height:1.5}header p{margin:0 0 18px;color:#526279}.controls{display:flex;align-items:center;gap:16px;flex-wrap:wrap;background:white;border:1px solid #dbe2ec;border-radius:10px;padding:14px;margin-bottom:18px}select,button{font:inherit;background:white;border:1px solid #aab8c9;border-radius:6px;padding:7px}input[type=range]{width:260px;vertical-align:middle;accent-color:#326fae}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:white;border:1px solid #dbe2ec;border-radius:10px;padding:15px;min-width:0}h2{font-size:16px;margin:0 0 8px}canvas{display:block;width:100%;height:auto}.caption{font-size:13px;color:#526279;margin:8px 0 0}.diagnostics{margin-top:16px;display:grid;grid-template-columns:1fr 1fr;gap:16px}table{border-collapse:collapse;font-size:13px;width:100%}th,td{text-align:right;padding:8px;border-bottom:1px solid #e2e7ee}th:first-child,td:first-child{text-align:left}.selected{background:#edf4fb}.legend{font-size:13px;color:#526279;margin-bottom:16px}.heatmap{display:grid;gap:2px;max-width:340px}.cell{padding:8px;text-align:center;font-size:12px}.metrics{font-size:13px;color:#526279;line-height:1.6}footer{font-size:12px;color:#526279;margin:20px 0}@media(max-width:760px){body{padding:12px}.grid,.diagnostics{grid-template-columns:1fr}input[type=range]{width:180px}}
</style></head><body><main>
<header><h1>__TITLE__</h1><p>Compare empirical couplings and the learned joint generator through the same frozen marginal flows.</p></header>
<div class="controls"><label>Coupling <select id="method" aria-label="Coupling method"></select></label><label>Marginal flow time <input id="time" type="range" min="0" max="__MAX_TIME__" value="__MID_TIME__" step="1" aria-label="Marginal flow time"> <output id="timeLabel"></output></label><button id="play" type="button">Play</button></div>
<div class="legend">● X &nbsp; △ Y &nbsp; ··· dashed line: paired observations &nbsp; · same color: the same pair across panels</div>
<div class="grid">
<section class="panel"><h2>Coupled outputs</h2><canvas id="outputs" width="560" height="450" aria-label="Coupled outputs in data coordinates"></canvas><p class="caption">Endpoint connections show the selected coupling; they are not ODE trajectories.</p></section>
<section class="panel"><h2>Reverse-mapped marginal noise</h2><canvas id="noise" width="560" height="450" aria-label="Marginal noise, inverse of frozen 2D flow"></canvas><p class="caption">Marginal noise (inverse of frozen 2D flow). These coordinates differ from the independent 4D source of the joint generator.</p></section>
<section class="panel"><h2>X marginal velocity and trajectories</h2><canvas id="fieldX" width="560" height="450" aria-label="Frozen X marginal velocity field"></canvas><p class="caption">Actual frozen 2D marginal velocity. Paths run from marginal noise toward outputs; arrows use a fixed scale across time.</p></section>
<section class="panel"><h2>Y marginal velocity and trajectories</h2><canvas id="fieldY" width="560" height="450" aria-label="Frozen Y marginal velocity field"></canvas><p class="caption">The same pair colors are retained through reverse mapping and marginal trajectories.</p></section>
</div>
<div class="diagnostics"><section class="panel"><h2>Component pairing</h2><p class="caption">Rows: X component. Columns: Y component. Cell values are joint probabilities.</p><div id="heatmap" class="heatmap"></div></section><section class="panel"><h2>All-pair evaluation</h2><div id="scores" style="overflow-x:auto"></div><p class="caption">SW: sliced Wasserstein marginal discrepancy, approximated with random projections; lower is better. Off support: fraction outside the target support.</p><p id="roundTrip" class="metrics"></p><p class="caption">Plots show a fixed subset. Metrics use the full evaluation population, when provided in the report.</p></section></div>
<footer>Offline demo: all arrays and drawing code are embedded in this file. The marginal time slider does not represent the 4D joint generator's sampling time.</footer>
</main><script type="application/json" id="demoData">__DATA__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("demoData").textContent);
const methods=data.method_names, methodSelect=document.getElementById("method"), timeInput=document.getElementById("time");
const labels=data.labels;
methods.forEach((name,index)=>{const option=document.createElement("option");option.value=String(index);option.textContent=labels[name];methodSelect.appendChild(option);});
methodSelect.value=String(Math.max(0,methods.indexOf("learned")));
const colors=data.output_x[0].map((_,i)=>`hsl(${(i*137.508)%360},65%,43%)`);
function layout(canvas,bounds){
 const ctx=canvas.getContext("2d"),padding=40,w=canvas.width,h=canvas.height;
 ctx.clearRect(0,0,w,h);ctx.fillStyle="#ffffff";ctx.fillRect(0,0,w,h);
 const scale=Math.min((w-2*padding)/(bounds[1]-bounds[0]),(h-2*padding)/(bounds[3]-bounds[2]));
 const cx=(bounds[0]+bounds[1])/2,cy=(bounds[2]+bounds[3])/2;
 const point=p=>[w/2+(p[0]-cx)*scale,h/2-(p[1]-cy)*scale];
 ctx.strokeStyle="#e3e9f1";ctx.lineWidth=1;ctx.font="11px system-ui";ctx.fillStyle="#607087";
 for(let j=0;j<=4;j++){
  const x=bounds[0]+j*(bounds[1]-bounds[0])/4,y=bounds[2]+j*(bounds[3]-bounds[2])/4;
  const a=point([x,bounds[2]]),b=point([x,bounds[3]]),c=point([bounds[0],y]),d=point([bounds[1],y]);
  ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.beginPath();ctx.moveTo(...c);ctx.lineTo(...d);ctx.stroke();
  ctx.textAlign="center";ctx.fillText(x.toFixed(1),a[0],a[1]+17);ctx.textAlign="right";ctx.fillText(y.toFixed(1),c[0]-7,c[1]+4);
 }
 ctx.textAlign="center";ctx.fillText("Coordinate 1",w/2,h-3);
 ctx.save();ctx.translate(10,h/2);ctx.rotate(-Math.PI/2);ctx.fillText("Coordinate 2",0,0);ctx.restore();
 return {ctx,point,scale};
}
function marker(ctx,p,color,triangle=false,size=4){
 ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.5;ctx.beginPath();
 if(triangle){ctx.moveTo(p[0],p[1]-size-1);ctx.lineTo(p[0]-size,p[1]+size);ctx.lineTo(p[0]+size,p[1]+size);ctx.closePath();ctx.stroke();}
 else{ctx.arc(p[0],p[1],size,0,Math.PI*2);ctx.fill();}
}
function pairs(canvas,x,y,bounds,showReference=false){
 const {ctx,point}=layout(canvas,bounds);
 if(showReference){ctx.fillStyle="#8995a6";ctx.globalAlpha=.16;for(const key of ["reference_x","reference_y"]){for(const p of data[key]??[]){const q=point(p);ctx.beginPath();ctx.arc(q[0],q[1],1.5,0,Math.PI*2);ctx.fill();}}ctx.globalAlpha=1;}
 x.forEach((p,i)=>{const a=point(p),b=point(y[i]);ctx.strokeStyle=colors[i];ctx.globalAlpha=.38;ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;marker(ctx,a,colors[i]);marker(ctx,b,colors[i],true);});
}
function field(canvas,side,m,t){
 const {ctx,point,scale}=layout(canvas,data.bounds),values=data["field_"+side][t];
 values.forEach((v,i)=>{const a=point(data.field_points[i]),vx=v[0]*scale/data.field_scale[side],vy=-v[1]*scale/data.field_scale[side],b=[a[0]+vx,a[1]+vy],angle=Math.atan2(vy,vx);
  ctx.strokeStyle="#536983";ctx.globalAlpha=.7;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);if(Math.hypot(vx,vy)>1){ctx.moveTo(b[0]-4*Math.cos(angle-.5),b[1]-4*Math.sin(angle-.5));ctx.lineTo(...b);ctx.lineTo(b[0]-4*Math.cos(angle+.5),b[1]-4*Math.sin(angle+.5));}ctx.stroke();
 });
 const paths=data["marginal_path_"+side][m];ctx.globalAlpha=.65;
 for(let i=0;i<Math.min(colors.length,24);i++){ctx.strokeStyle=colors[i];ctx.lineWidth=1.3;ctx.beginPath();for(let j=0;j<=t;j++){const p=point(paths[j][i]);j===0?ctx.moveTo(...p):ctx.lineTo(...p);}ctx.stroke();}
 ctx.globalAlpha=1;paths[t].forEach((p,i)=>marker(ctx,point(p),colors[i],side==="y",3));
}
function reward(index){const metrics=data.report.methods?.[methods[index]]??{};for(const key of ["mismatch_reward","component_mismatch","different_component_probability"]){if(Number.isFinite(metrics[key]))return metrics[key];}const mat=data.component_pair_matrices[index];return mat.reduce((sum,row,i)=>sum+row.reduce((s,v,j)=>s+(i===j?0:v),0),0);}
function diagnostics(m){
 const heatmap=document.getElementById("heatmap"),matrix=data.component_pair_matrices[m],maximum=Math.max(...data.component_pair_matrices.flat(2),1e-10);heatmap.replaceChildren();heatmap.style.gridTemplateColumns=`repeat(${matrix.length},1fr)`;
 matrix.forEach((row,i)=>row.forEach((value,j)=>{const cell=document.createElement("div"),shade=value/maximum;cell.className="cell";cell.style.background=`hsl(211,65%,${97-shade*60}%)`;cell.style.color=shade>.55?"white":"#17233a";cell.textContent=value.toFixed(3);cell.title=`X component ${i}, Y component ${j}: ${value.toFixed(5)}`;heatmap.appendChild(cell);}));
 const table=document.createElement("table"),head=document.createElement("tr");["Method","Different components","SW X","SW Y","Off support X","Off support Y"].forEach(text=>{const th=document.createElement("th");th.textContent=text;head.appendChild(th);});table.appendChild(head);
 methods.forEach((name,i)=>{const row=document.createElement("tr");if(i===m)row.className="selected";const label=document.createElement("td"),value=document.createElement("td");label.textContent=labels[name];value.textContent=reward(i).toFixed(4);row.append(label,value);const metrics=data.report.methods?.[name]??{};for(const key of ["x_sliced_wasserstein","y_sliced_wasserstein","x_off_support_mass","y_off_support_mass"]){const cell=document.createElement("td");cell.textContent=Number.isFinite(metrics[key])?metrics[key].toFixed(4):"—";row.appendChild(cell);}table.appendChild(row);});document.getElementById("scores").replaceChildren(table);
 const round=data.report.round_trip?.[methods[m]];document.getElementById("roundTrip").textContent=round?`Inverse → forward reconstruction RMSE: X ${Number(round.x_rmse).toExponential(2)}, Y ${Number(round.y_rmse).toExponential(2)}.`:"";
}
function render(){const m=Number(methodSelect.value),t=Number(timeInput.value);document.getElementById("timeLabel").textContent=data.times[t].toFixed(2);pairs(document.getElementById("outputs"),data.output_x[m],data.output_y[m],data.bounds,true);pairs(document.getElementById("noise"),data.noise_x[m],data.noise_y[m],data.noise_bounds);field(document.getElementById("fieldX"),"x",m,t);field(document.getElementById("fieldY"),"y",m,t);diagnostics(m);}
let timer=null;document.getElementById("play").addEventListener("click",()=>{if(timer){clearInterval(timer);timer=null;document.getElementById("play").textContent="Play";}else{document.getElementById("play").textContent="Pause";timer=setInterval(()=>{timeInput.value=String((Number(timeInput.value)+1)%data.times.length);render();},120);}});
methodSelect.addEventListener("change",render);timeInput.addEventListener("input",render);render();
</script></body></html>
"""


def _interactive(output_dir: Path, data: dict[str, np.ndarray], report: dict[str, Any], title: str) -> Path:
    payload = {key: value.tolist() for key, value in data.items() if key not in {"joint_path", "ordinary_path", "ordinary_linear_path"}}
    for key in ("reference_x", "reference_y"):
        if key in data:
            indices = np.linspace(0, len(data[key]) - 1, min(len(data[key]), 1000), dtype=int)
            payload[key] = data[key][indices].tolist()
    payload["labels"] = {name: _label(name) for name in data["method_names"].tolist()}
    payload["report"] = report
    payload["noise_bounds"] = _point_bounds(data["noise_x"], data["noise_y"]).tolist()
    span = max(float(data["bounds"][1] - data["bounds"][0]), float(data["bounds"][3] - data["bounds"][2]))
    payload["field_scale"] = {
        side: max(float(np.quantile(np.linalg.norm(data[f"field_{side}"], axis=-1), 0.95)) / (0.075 * span), 1e-8)
        for side in ("x", "y")
    }
    # Escape HTML delimiters even inside the JSON script block. Values are later
    # inserted with textContent rather than HTML interpolation.
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    serialized = serialized.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    page = _HTML.replace("__TITLE__", html.escape(title)).replace("__MAX_TIME__", str(len(data["times"]) - 1)).replace("__MID_TIME__", str(int(np.argmin(np.abs(data["times"] - 0.5))))).replace("__DATA__", serialized)
    path = output_dir / "demo.html"
    path.write_text(page, encoding="utf-8")
    return path


def render_demo(output_dir: str | Path, *, title: str | None = None) -> list[Path]:
    """Render ``display.npz`` and ``report.json`` without retraining any model.

    Returns the overview figure, quantitative diagnostics, an optional ordinary
    transport figure, and self-contained interactive HTML. Main display paths
    run in forward marginal time, including inverse trajectories subsequently
    reversed. Optional ordinary paths instead start in the source data measure.
    """
    output_dir = Path(output_dir)
    data = _load_display(output_dir / "display.npz")
    report_path = output_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if title is None:
        experiment = report.get("experiment", {})
        title = experiment.get("title", experiment.get("name", "Diverse coupling experiment")) if isinstance(experiment, dict) else str(experiment)
    figures = _figures(output_dir, data, report, str(title))
    return [*figures, _interactive(output_dir, data, report, str(title))]
