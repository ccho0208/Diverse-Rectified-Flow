"""Train frozen marginals, optimize empirical pairs, and export inspectable demos."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import platform
import signal
import time

import numpy as np
import scipy
import torch

from ..costs import build_reward_matrix
from ..coupling import CouplingPlan, solve_optimal_coupling
from .config import DemoConfig
from .datasets import build_spec
from .flows import FlowTrainingConfig, integrate, train_flow, velocity_grid
from .metrics import evaluate_pairs, empirical_independent_matrix
from .visualize import render_demo


def _json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch environment cannot access a GPU")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return str(device)


def _random_sources(seed: int) -> list[np.random.Generator]:
    return [np.random.default_rng(stream) for stream in np.random.SeedSequence(seed).spawn(8)]


def _model_identity(model) -> str:
    digest = hashlib.sha256(json.dumps(model.export_config(), sort_keys=True).encode())
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _train_config(config: DemoConfig, epochs: int, seed: int) -> FlowTrainingConfig:
    return FlowTrainingConfig(
        epochs=epochs, learning_rate=config.learning_rate, seed=seed,
        device=config.device, dtype=config.dtype, chunk_size=config.chunk_size,
        checkpoint_every=config.checkpoint_every)


def _progress(experiment: str, role: str, checkpoint_every: int):
    def emit(record: dict):
        if record["epoch"] == 1 or record["epoch"] % min(checkpoint_every, 100) == 0:
            print(json.dumps({"experiment": experiment, "model": role, **record}), flush=True)
    return emit


def _forward(model, points, config, *, record=False, reverse=False):
    return integrate(
        model, points, steps=config.ode_steps,
        start_time=1.0 if reverse else 0.0, end_time=0.0 if reverse else 1.0,
        record_trajectory=record, batch_size=config.sampling_batch_size,
        device=config.device)


def _chosen_reward(metrics: dict, config: DemoConfig) -> float:
    return metrics["mismatch_reward" if config.reward == "component_mismatch"
                   else "squared_distance_reward"]


def run_experiment(config: DemoConfig, name: str, output: Path, *, resume: bool = False,
                   stop_requested=None) -> dict:
    started = time.monotonic()
    spec = build_spec(name, ring_components=config.ring_components)
    output.mkdir(parents=True, exist_ok=True)
    seed = config.seed + {"gmm8": 0, "two_disks": 10000, "rings": 20000}[name]
    rng_data, rng_pool, rng_eval, rng_iid, rng_display, rng_joint, rng_transport, _ = _random_sources(seed)
    disk_setup = name == "two_disks"
    marginal_source = spec.source_distribution()
    joint_source = spec.source_distribution(copies=2)

    def flow(model, points, *, record=False, reverse=False):
        if stop_requested is not None and stop_requested():
            raise InterruptedError("stop requested; continue from checkpoints with --resume")
        return _forward(model, points, config, record=record, reverse=reverse)

    dataset_path = output / "datasets.npz"
    if resume and dataset_path.exists():
        data = _archive(dataset_path)
        if disk_setup and "marginal_sources" not in data:
            raise ValueError("old two_disks Gaussian-source setup cannot be resumed; choose a new run directory")
    else:
        target_x, labels_x = spec.sample(config.marginal_samples, rng_data, side="x")
        target_y, labels_y = target_x.copy(), labels_x.copy()
        reference_x, _ = spec.sample(config.evaluation_samples, rng_eval, side="x")
        reference_y, _ = spec.sample(config.evaluation_samples, rng_eval, side="y")
        data = {"target_x": target_x, "target_y": target_y,
                "target_labels_x": labels_x, "target_labels_y": labels_y,
                "reference_x": reference_x, "reference_y": reference_y,
                "pool_noise": marginal_source.sample_numpy(config.training_samples, 2, rng_pool),
                "joint_noise": joint_source.sample_numpy(config.evaluation_samples, 4, rng_joint)}
        if disk_setup:
            data["marginal_sources"], _ = spec.sample(config.marginal_samples, rng_data, side="source")
            data["source_reference"], _ = spec.sample(config.evaluation_samples, rng_eval, side="source")
        _npz(dataset_path, **data)
    _json(output / "config.json", {"experiment": spec.to_dict(), "run": config.to_dict()})

    def train(role, targets, epochs, role_seed, *, masses=None, source_points=None,
              source_distribution=None):
        folder = output / role
        return train_flow(
            targets, _train_config(config, epochs, role_seed), output_dir=folder,
            masses=masses, source_points=source_points, source_distribution=source_distribution,
            hidden_sizes=config.hidden_sizes,
            resume=resume and (folder / "checkpoint.pt").exists(),
            metadata={"experiment": spec.to_dict(), "role": role,
                      "reward": config.reward, "shared_unconstrained_architecture": True},
            progress=_progress(name, role, config.checkpoint_every),
            stop_requested=stop_requested)

    print(json.dumps({"experiment": name, "stage": "marginal_training"}), flush=True)
    model_x = train("marginal_x", data["target_x"], config.marginal_epochs, seed + 1,
                    source_points=data.get("marginal_sources"))
    model_y = model_x
    antithetic_noise = marginal_source.antithetic(data["pool_noise"])

    if stop_requested is not None and stop_requested():
        raise InterruptedError("stop requested; marginal checkpoints saved")
    identities = (_model_identity(model_x), _model_identity(model_y))
    cached_pool = _archive(output / "pool.npz") if resume and (output / "pool.npz").exists() else None
    reuse_pool = (cached_pool is not None
                  and str(cached_pool.get("model_x_identity", "")) == identities[0]
                  and str(cached_pool.get("model_y_identity", "")) == identities[1])
    if reuse_pool:
        # Preserve exact empirical targets across resumed CPU/GPU environments.
        X, Y = cached_pool["X"], cached_pool["Y"]
    else:
        X = flow(model_x, data["pool_noise"]).endpoints
        Y = flow(model_y, antithetic_noise).endpoints
        if cached_pool is not None and (output / "joint").exists():
            # Extended marginal training changes the joint training measure.
            # Preserve the previous joint model before starting a fresh one.
            archived = output / "previous_joint_models"
            archived.mkdir(exist_ok=True)
            destination = archived / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            (output / "joint").rename(destination)
    x_labels, y_labels = spec.classify(X, side="x"), spec.classify(Y, side="y")
    C = build_reward_matrix(X, Y, config.reward, x_labels=x_labels, y_labels=y_labels)
    plan = (CouplingPlan.load(output / "plan.npz")
            if reuse_pool and (output / "plan.npz").exists()
            else solve_optimal_coupling(C, use_permutation=config.use_permutation))
    plan.save(output / "plan.npz")
    np.save(output / "C.npy", C, allow_pickle=False)
    _npz(output / "pool.npz", X=X, Y=Y, x_labels=x_labels, y_labels=y_labels,
         noise_x=data["pool_noise"], noise_y=antithetic_noise,
         x_ids=np.arange(len(X)), y_ids=np.arange(len(Y)),
         model_x_identity=np.asarray(identities[0]), model_y_identity=np.asarray(identities[1]))
    joint_targets = np.concatenate((X[plan.row_indices], Y[plan.column_indices]), axis=1)
    print(json.dumps({"experiment": name, "stage": "coupling", "reward": plan.reward,
                      "solver": plan.solver, "nonzero_pairs": len(plan.masses)}), flush=True)
    joint = train("joint", joint_targets, config.joint_epochs, seed + 3, masses=plan.masses,
                  source_distribution=joint_source)
    if stop_requested is not None and stop_requested():
        raise InterruptedError("stop requested; joint checkpoint saved")
    sampled = flow(joint, data["joint_noise"]).endpoints
    generated_x, generated_y = sampled[:, :2], sampled[:, 2:]
    reference = {"reference_X": data["reference_x"], "reference_Y": data["reference_y"],
                 "seed": seed + 55}
    methods = {
        "empirical_optimal": evaluate_pairs(
            X[plan.row_indices], Y[plan.column_indices], spec, weights=plan.masses, **reference),
        "antithetic": evaluate_pairs(X, Y, spec, **reference),
        "learned": evaluate_pairs(generated_x, generated_y, spec, **reference),
    }
    # The independent empirical baseline is analytic. Monte Carlo pairs are
    # used only for marginal diagnostics and display, not its exact reward/table.
    independent_rows = rng_iid.integers(len(X), size=config.evaluation_samples)
    independent_cols = rng_iid.integers(len(Y), size=config.evaluation_samples)
    independent = evaluate_pairs(X[independent_rows], Y[independent_cols], spec, **reference)
    independent_matrix = empirical_independent_matrix(X, Y, spec)
    independent["component_pair_matrix"] = independent_matrix.tolist()
    independent["mismatch_reward"] = float(1 - np.trace(independent_matrix))
    independent["squared_distance_reward"] = float(
        np.mean(np.sum(X.astype(np.float64) ** 2, axis=1))
        + np.mean(np.sum(Y.astype(np.float64) ** 2, axis=1))
        - 2 * np.dot(X.mean(axis=0, dtype=np.float64), Y.mean(axis=0, dtype=np.float64)))
    independent["x_component_frequencies"] = independent_matrix.sum(axis=1).tolist()
    independent["y_component_frequencies"] = independent_matrix.sum(axis=0).tolist()
    methods["independent"] = independent

    transport_model = None
    ordinary_path = None
    ordinary_metrics = None
    if name == "two_disks" and config.ordinary_transport:
        # The frozen marginal is itself the conventional left-to-right RF.
        # Reuse it so the picture and the endpoint experiment use exactly F.
        source = data["marginal_sources"]
        target = data["target_x"]
        transport_model = model_x
        transport_source, _ = spec.sample(config.evaluation_samples, rng_transport, side="source")
        transported = flow(transport_model, transport_source).endpoints
        ordinary_metrics = evaluate_pairs(
            transport_source, transported, spec, x_side="source",
            reference_X=data["source_reference"], reference_Y=data["reference_y"], seed=seed + 55)

    # Display a fixed subset. Inverse paths use the frozen marginal models, and
    # are distinct from the independent 4D source of the joint model.
    count = config.display_samples
    rows = rng_display.choice(len(plan.masses), count, replace=False)
    display_outputs = {
        "empirical_optimal": (X[plan.row_indices[rows]], Y[plan.column_indices[rows]]),
        "antithetic": (X[:count], Y[:count]),
        "independent": (X[independent_rows[:count]], Y[independent_cols[:count]]),
        "learned": (generated_x[:count], generated_y[:count]),
    }
    if transport_model is not None:
        ordinary_result = flow(transport_model, source[:count], record=True)
        ordinary_path = ordinary_result.trajectory
    frames = np.linspace(0, config.ode_steps, config.display_frames, dtype=int)
    names = list(display_outputs)
    paths_x, paths_y, noises_x, noises_y, round_trips = [], [], [], [], {}
    for method, (a, b) in display_outputs.items():
        inverse_x = flow(model_x, a, record=True, reverse=True)
        inverse_y = flow(model_y, b, record=True, reverse=True)
        recovered_x = flow(model_x, inverse_x.endpoints).endpoints
        recovered_y = flow(model_y, inverse_y.endpoints).endpoints
        round_trips[method] = {
            "x_rmse": float(np.sqrt(np.mean((recovered_x - a) ** 2))),
            "y_rmse": float(np.sqrt(np.mean((recovered_y - b) ** 2))),
            "samples": count, "definition": "root mean squared error per coordinate"}
        paths_x.append(inverse_x.trajectory[::-1][frames])
        paths_y.append(inverse_y.trajectory[::-1][frames])
        noises_x.append(inverse_x.endpoints)
        noises_y.append(inverse_y.endpoints)
    learned_noise_x = flow(model_x, generated_x, reverse=True).endpoints
    learned_noise_y = flow(model_y, generated_y, reverse=True).endpoints
    reconstructed_x = flow(model_x, learned_noise_x).endpoints
    reconstructed_y = flow(model_y, learned_noise_y).endpoints
    round_trips["learned"] = {
        "x_rmse": float(np.sqrt(np.mean((reconstructed_x - generated_x) ** 2))),
        "y_rmse": float(np.sqrt(np.mean((reconstructed_y - generated_y) ** 2))),
        "samples": len(generated_x), "definition": "root mean squared error per coordinate",
    }
    _npz(output / "samples.npz", X=generated_x, Y=generated_y, joint=sampled,
         initial_joint_noise=data["joint_noise"],
         marginal_noise_x=learned_noise_x, marginal_noise_y=learned_noise_y,
         pair_ids=np.arange(len(generated_x)))
    times = np.linspace(0, 1, config.ode_steps + 1)[frames]
    bounds = np.asarray(spec.bounds(), dtype=np.float64)
    if not disk_setup:
        bounds[[0, 2]] = np.minimum(bounds[[0, 2]], -3.5)
        bounds[[1, 3]] = np.maximum(bounds[[1, 3]], 3.5)
    field_x = velocity_grid(model_x, bounds, times, grid_size=config.grid_size, device=config.device)
    field_y = velocity_grid(model_y, bounds, times, grid_size=config.grid_size, device=config.device)
    joint_path = flow(joint, data["joint_noise"][:count], record=True).trajectory[frames]
    display = {
        "times": times, "field_points": field_x["points"],
        "field_x": field_x["velocities"], "field_y": field_y["velocities"],
        "bounds": bounds, "method_names": np.asarray(names),
        "output_x": np.stack([display_outputs[m][0] for m in names]),
        "output_y": np.stack([display_outputs[m][1] for m in names]),
        "noise_x": np.stack(noises_x), "noise_y": np.stack(noises_y),
        "marginal_path_x": np.stack(paths_x), "marginal_path_y": np.stack(paths_y),
        "component_pair_matrices": np.asarray([methods[m]["component_pair_matrix"] for m in names]),
        "joint_path": joint_path,
        "reference_x": data["reference_x"][:min(512, len(data["reference_x"]))],
        "reference_y": data["reference_y"][:min(512, len(data["reference_y"]))],
    }
    if ordinary_path is not None:
        display["ordinary_path"] = ordinary_path[frames]
        display["ordinary_linear_path"] = (
            (1 - times[:, None, None]) * source[:count]
            + times[:, None, None] * target[:count])
    if disk_setup:
        display["source_reference"] = data["source_reference"][:512]
    _npz(output / "display.npz", **display)
    _npz(output / "reverse_mapping.npz", method_names=display["method_names"],
         noise_x=display["noise_x"], noise_y=display["noise_y"],
         output_x=display["output_x"], output_y=display["output_y"],
         times=times, path_x=display["marginal_path_x"], path_y=display["marginal_path_y"],
         learned_noise_x=learned_noise_x, learned_noise_y=learned_noise_y,
         learned_pair_ids=np.arange(len(generated_x)))
    report = {
        "experiment": spec.to_dict(), "methods": methods, "round_trip": round_trips,
        "empirical_optimization": {"reward": config.reward, "value": plan.reward,
                                   "solver": plan.solver, "diagnostics": plan.diagnostics,
                                   "independent_value": float(C.mean()),
                                   "antithetic_value": float(np.trace(C) / len(C))},
        "learned_reward_difference_from_empirical_optimum": float(
            plan.reward - _chosen_reward(methods["learned"], config)),
        "notes": ["Empirical comparisons use the same frozen-flow endpoint pools.",
                  "Learned samples have approximate marginals; their reward difference is not an optimality certificate.",
                  "All flows use the same unconstrained time-conditioned MLP; ring symmetry is not enforced.",
                  "Reverse noise is the inverse of each frozen 2D marginal flow, distinct from initial 4D joint noise."],
        "duration_seconds": time.monotonic() - started,
    }
    report["flow_setup"] = {
        "marginal_source": marginal_source.to_dict(), "joint_source": joint_source.to_dict(),
        "shared_marginal": True, "coupling_space": "two target-distribution endpoints",
        "antithetic_transform": "(-4 - x, -y)" if disk_setup else "-z",
    }
    if ordinary_metrics is not None:
        report["ordinary_transport"] = ordinary_metrics
    if disk_setup:
        report["reverse_source"] = evaluate_pairs(
            learned_noise_x, learned_noise_y, spec, x_side="source", y_side="source",
            reference_X=data["source_reference"], reference_Y=data["source_reference"], seed=seed + 56)
    _json(output / "report.json", report)
    render_demo(output, title={"gmm8": "Eight Gaussian components", "two_disks": "Two uniform disks",
                               "rings": f"{spec.components} concentric rings"}[name])
    print(json.dumps({"experiment": name, "stage": "complete", "output": str(output),
                      "seconds": report["duration_seconds"]}), flush=True)
    return report


def run_demos(config: DemoConfig, output: str | Path, *, resume: bool = False) -> dict:
    config.validate()
    config = replace(config, device=resolve_device(config.device))
    if config.device.startswith("mps") and config.dtype == "float64":
        raise ValueError("MPS requires float32")
    torch.set_num_threads(config.cpu_threads)
    torch.set_float32_matmul_precision("high")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old_path = output / "config.json"
    setup_path = output / "setup.json"
    if old_path.exists():
        if not resume:
            raise ValueError("run already exists; use --resume or choose a new output directory")
        if "two_disks" in config.experiments:
            setup = json.loads(setup_path.read_text()) if setup_path.exists() else {}
            if setup.get("two_disks") != "left_disks_to_right_disks_v2":
                raise ValueError("old two_disks Gaussian-source setup cannot be resumed; choose a new run directory")
        previous = json.loads(old_path.read_text())
        changeable = {"marginal_epochs", "joint_epochs", "transport_epochs", "device", "cpu_threads"}
        for key, value in config.to_dict().items():
            if key not in changeable and previous.get(key) != value:
                raise ValueError(f"cannot change {key} when resuming; choose a new run directory")
    elif any(output.iterdir()):
        raise ValueError("output must be empty unless resuming an existing demo run")
    _json(old_path, config.to_dict())
    _json(setup_path, {name: "left_disks_to_right_disks_v2" if name == "two_disks" else "gaussian_source_v1"
                       for name in config.experiments})
    environment = {
        "time_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "device": config.device,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(config.device)) if config.device.startswith("cuda") else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _json(output / "environment.json", environment)
    stop = {"requested": False}
    previous_handlers = {}

    def request_stop(signum, frame):
        stop["requested"] = True
        print(json.dumps({"stage": "checkpoint_stop_requested", "signal": signum}), flush=True)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", None)):
            if sig is not None:
                previous_handlers[sig] = signal.signal(sig, request_stop)
        reports = {}
        for name in config.experiments:
            if stop["requested"]:
                raise InterruptedError("stop requested; continue with --resume")
            reports[name] = run_experiment(config, name, output / name, resume=resume,
                                           stop_requested=lambda: stop["requested"])
        summary = {"config": config.to_dict(), "experiments": {
            name: {"report": f"{name}/report.json", "demo": f"{name}/demo.html",
                   "empirical_reward": report["empirical_optimization"]["value"],
                   "learned_mismatch_reward": report["methods"]["learned"]["mismatch_reward"]}
            for name, report in reports.items()}}
        _json(output / "summary.json", summary)
        return summary
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
