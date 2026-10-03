from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from diverse_coupling.demos.config import DemoConfig
from diverse_coupling.demos.datasets import build_spec
from diverse_coupling.demos.flows import load_flow
from diverse_coupling.demos.runner import run_demos


def tiny_config(**kwargs):
    config = DemoConfig(training_samples=16, marginal_samples=32, evaluation_samples=32,
                        marginal_epochs=2, joint_epochs=2, transport_epochs=2,
                        chunk_size=16, checkpoint_every=1, ode_steps=4,
                        display_samples=4, display_frames=3, grid_size=3,
                        device="cpu", cpu_threads=1)
    return replace(config, **kwargs)


class DemoRunnerTests(unittest.TestCase):
    def test_all_experiments_have_exact_baselines_reverse_paths_and_fields(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            summary = run_demos(tiny_config(), root)
            self.assertEqual(set(summary["experiments"]), {"gmm8", "two_disks", "rings"})
            for name in summary["experiments"]:
                folder = root / name
                report = json.loads((folder / "report.json").read_text())
                C = np.load(folder / "C.npy")
                optimization = report["empirical_optimization"]
                self.assertAlmostEqual(optimization["independent_value"], C.mean())
                self.assertAlmostEqual(optimization["antithetic_value"], np.trace(C) / len(C))
                self.assertGreaterEqual(optimization["value"] + 1e-9, optimization["independent_value"])
                self.assertGreaterEqual(optimization["value"] + 1e-9, optimization["antithetic_value"])
                self.assertTrue((folder / "demo.html").exists())
                self.assertTrue((folder / "overview.png").exists())
                with np.load(folder / "samples.npz", allow_pickle=False) as samples:
                    self.assertEqual(samples["marginal_noise_x"].shape, (32, 2))
                    self.assertEqual(samples["marginal_noise_y"].shape, (32, 2))
                    np.testing.assert_array_equal(samples["pair_ids"], np.arange(32))
                with np.load(folder / "display.npz", allow_pickle=False) as data:
                    np.testing.assert_allclose(data["marginal_path_x"][:, 0], data["noise_x"])
                    np.testing.assert_allclose(data["marginal_path_y"][:, 0], data["noise_y"])
                    np.testing.assert_allclose(data["marginal_path_x"][:, -1], data["output_x"])
                    np.testing.assert_allclose(data["marginal_path_y"][:, -1], data["output_y"])
                    model, payload = load_flow(folder / "marginal_x" / "checkpoint.pt")
                    self.assertEqual(payload["model_config"]["hidden_sizes"], [64, 64, 64])
                    with torch.no_grad():
                        expected = model(torch.tensor(data["field_points"], dtype=torch.float32),
                                         torch.tensor(float(data["times"][1])))
                    np.testing.assert_allclose(data["field_x"][1], expected.numpy(), atol=1e-6)
                    learned = data["method_names"].tolist().index("learned")
                    with np.load(folder / "samples.npz", allow_pickle=False) as samples:
                        # Float32 matrix kernels can round differently for a
                        # display batch and the full evaluation batch.
                        np.testing.assert_allclose(data["noise_x"][learned], samples["marginal_noise_x"][:4],
                                                   rtol=1e-5, atol=1e-6)
                        np.testing.assert_allclose(data["noise_y"][learned], samples["marginal_noise_y"][:4],
                                                   rtol=1e-5, atol=1e-6)
                    if name == "two_disks":
                        np.testing.assert_allclose(data["ordinary_linear_path"][0], data["ordinary_path"][0])
                        self.assertNotIn("ordinary_transport", data["method_names"].tolist())
                        self.assertIn("ordinary_transport", report)
                        spec = build_spec(name)
                        self.assertTrue(spec.support_mask(data["ordinary_path"][0], "source").all())
                        np.testing.assert_array_equal(data["field_x"], data["field_y"])
                if name == "two_disks":
                    self.assertFalse((folder / "marginal_y").exists())
                    with np.load(folder / "datasets.npz") as data:
                        for key in ("target_x", "target_y", "reference_x", "reference_y"):
                            self.assertTrue(spec.support_mask(data[key]).all())
                        self.assertTrue(spec.support_mask(data["marginal_sources"], "source").all())
                        for key, dimensions in (("pool_noise", 2), ("joint_noise", 4)):
                            self.assertEqual(data[key].shape[1], dimensions)
                            self.assertTrue(spec.support_mask(data[key].reshape(-1, 2), "source").all())
                    with np.load(folder / "pool.npz") as data:
                        np.testing.assert_allclose(data["noise_y"], [-4, 0] - data["noise_x"])
                    with np.load(folder / "samples.npz") as data:
                        self.assertTrue(spec.support_mask(data["initial_joint_noise"].reshape(-1, 2), "source").all())
                    self.assertIn("reverse_source", report)
                    _, marginal = load_flow(folder / "marginal_x/checkpoint.pt")
                    _, joint = load_flow(folder / "joint/checkpoint.pt")
                    self.assertEqual(marginal["source_kind"], "fixed")
                    self.assertEqual(joint["source_kind"], "uniform_disks")
                    self.assertEqual(joint["source_distribution"]["copies"], 2)
                    self.assertIn("Both coupled outputs lie in the right-disk target", (folder / "demo.html").read_text())

    def test_resume_preserves_pool_and_extends_joint_training(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = tiny_config(experiments=("rings",), ordinary_transport=False)
            run_demos(config, root / "resumed")
            with np.load(root / "resumed/rings/pool.npz") as stored:
                original_x = stored["X"].copy()
            run_demos(replace(config, joint_epochs=3), root / "resumed", resume=True)
            run_demos(replace(config, joint_epochs=3), root / "uninterrupted")
            with np.load(root / "resumed/rings/pool.npz") as stored:
                np.testing.assert_array_equal(stored["X"], original_x)
            a, payload = load_flow(root / "resumed/rings/joint/checkpoint.pt")
            b, _ = load_flow(root / "uninterrupted/rings/joint/checkpoint.pt")
            self.assertEqual(payload["completed_epochs"], 3)
            for p, q in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "training_samples"):
                run_demos(replace(config, training_samples=17), root / "resumed", resume=True)

    def test_extended_marginal_training_preserves_old_joint_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = tiny_config(experiments=("gmm8",), ordinary_transport=False)
            run_demos(config, root)
            old_model, _ = load_flow(root / "gmm8/joint/checkpoint.pt")
            run_demos(replace(config, marginal_epochs=3), root, resume=True)
            archives = list((root / "gmm8/previous_joint_models").iterdir())
            self.assertEqual(len(archives), 1)
            archived_model, _ = load_flow(archives[0] / "checkpoint.pt")
            for p, q in zip(old_model.parameters(), archived_model.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_config_rejects_invalid_geometry_and_unknown_keys(self):
        for config in (tiny_config(ring_components=3), tiny_config(display_frames=9),
                       tiny_config(use_permutation="yes"), tiny_config(grid_size=1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text('{"typo_samples": 42}')
            with self.assertRaisesRegex(ValueError, "unknown"):
                DemoConfig.load(path)

    def test_disk_resume_matches_uninterrupted_and_rejects_old_gaussian_setup(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = tiny_config(experiments=("two_disks",), ordinary_transport=False)
            run_demos(config, root / "resumed")
            run_demos(replace(config, joint_epochs=3), root / "resumed", resume=True)
            run_demos(replace(config, joint_epochs=3), root / "full")
            a, payload = load_flow(root / "resumed/two_disks/joint/checkpoint.pt")
            b, _ = load_flow(root / "full/two_disks/joint/checkpoint.pt")
            self.assertEqual(payload["source_kind"], "uniform_disks")
            for p, q in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            (root / "resumed/setup.json").unlink()
            old_config = (root / "resumed/config.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "old two_disks.*new run directory"):
                run_demos(config, root / "resumed", resume=True)
            self.assertEqual(old_config, (root / "resumed/config.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
