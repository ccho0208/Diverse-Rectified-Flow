from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from diverse_coupling.cli import main
from diverse_coupling.coupling import CouplingPlan, coupling_from_permutation
from diverse_coupling.data import build_paired_dataset
from diverse_coupling.models import RectifiedFlowAdapter
from diverse_coupling.sample import integrate_velocity, load_joint_model, sample_joint_model
from diverse_coupling.train import (
    TrainConfig, accumulate_plan_gradients, train_joint_model, weighted_objective,
)


def fractional_data():
    plan = CouplingPlan((2, 2), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]),
                        np.array([0.1, 0.4, 0.4, 0.1]))
    return build_paired_dataset(np.array([[1.0, -2.0], [3.0, 0.5]]),
                                np.array([-4.0, 2.0]), plan)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rf_loss_is_per_pair_squared_velocity_error(self):
        model = RectifiedFlowAdapter(3, (), seed=4).double()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        targets = torch.tensor([[1., 2., 3.], [-2., 0., 4.]], dtype=torch.float64)
        noise = torch.tensor([[2., 0., 1.], [1., 1., 0.]], dtype=torch.float64)
        times = torch.tensor([0.2, 0.8], dtype=torch.float64)
        loss = model.per_example_loss(targets, noise=noise, times=times)
        torch.testing.assert_close(loss, (targets - noise).square().sum(dim=1))
        self.assertEqual(loss.shape, (2,))

    def test_fractional_rf_gradients_match_dense_objective_and_chunks(self):
        data = fractional_data()
        noise = torch.linspace(-2, 3, 12, dtype=torch.float64).reshape(4, 3)
        times = torch.tensor([0.1, 0.3, 0.7, 0.9], dtype=torch.float64)
        reference = RectifiedFlowAdapter(3, (8,), seed=5).double()
        batch = data.get_batch(np.arange(4), dtype=torch.float64)
        expected = weighted_objective(
            reference.per_example_loss(batch.targets, noise=noise, times=times), batch.masses)
        expected.backward()
        for chunk_size in (None, 1, 3):
            with self.subTest(chunk_size=chunk_size):
                model = RectifiedFlowAdapter(3, (8,), seed=5).double()
                actual = accumulate_plan_gradients(
                    data, model, chunk_size=chunk_size, dtype=torch.float64,
                    noise=noise, times=times)
                self.assertAlmostEqual(actual, expected.item(), places=12)
                for p, q in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)

    def test_training_rng_and_optimizer_updates_preserve_chunk_equivalence(self):
        data = fractional_data()
        models = [RectifiedFlowAdapter(3, (8,), seed=12) for _ in range(2)]
        results = [train_joint_model(data, model, TrainConfig(
            epochs=3, seed=9, dtype="float64", chunk_size=chunk))
            for model, chunk in zip(models, (None, 1))]
        np.testing.assert_allclose(
            [r["loss"] for r in results[0].history],
            [r["loss"] for r in results[1].history], rtol=1e-12, atol=1e-12)
        for a, b in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        self.assertEqual(results[1].history[-1]["optimizer_steps"], 3)

    def test_minibatches_preserve_pairs_visit_once_and_weight_partial_batch(self):
        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.joint_dim = 2
                self.anchor = nn.Parameter(torch.zeros(()))
                self.seen = []

            def per_example_loss(self, targets, *, noise, times):
                self.seen.extend(targets.detach().tolist())
                return targets.square().sum(dim=1) + self.anchor * 0

        sigma = np.array([3, 0, 4, 1, 2])
        data = build_paired_dataset(np.arange(5.), np.arange(5.) + 10,
                                    coupling_from_permutation(sigma))
        model = Recorder()
        result = train_joint_model(data, model, TrainConfig(
            epochs=2, mode="permutation_minibatch", batch_size=2, dtype="float64"))
        expected_pairs = [[float(i), float(10 + sigma[i])] for i in range(5)]
        for epoch in range(2):
            self.assertEqual(sorted(model.seen[epoch * 5:(epoch + 1) * 5]), expected_pairs)
        expected_loss = np.mean([x * x + y * y for x, y in expected_pairs])
        for item in result.history:
            self.assertAlmostEqual(item["loss"], expected_loss, places=12)
        self.assertEqual(result.history[-1]["optimizer_steps"], 6)

    def test_checkpoint_reload_seeded_sampling_and_trajectory_shapes(self):
        data = fractional_data()
        model = RectifiedFlowAdapter(3, (8,), seed=8)
        with tempfile.TemporaryDirectory() as temporary:
            result = train_joint_model(data, model, TrainConfig(epochs=4, chunk_size=3),
                                       output_dir=temporary, metadata={"purpose": "test"})
            self.assertTrue(all(np.isfinite(item["loss"]) for item in result.history))
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
            loaded, checkpoint = load_joint_model(result.checkpoint_path)
            self.assertEqual(checkpoint["metadata"], {"purpose": "test"})
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
            sampled = sample_joint_model(result.checkpoint_path, 7, True, steps=5, seed=3)
            repeated = sample_joint_model(result.checkpoint_path, 7, True, steps=5, seed=3)
            self.assertEqual(sampled.X.shape, (7, 2))
            self.assertEqual(sampled.Y.shape, (7,))
            self.assertEqual(sampled.trajectory.shape, (6, 7, 3))
            self.assertEqual(sampled.x_trajectory.shape, (6, 7, 2))
            self.assertEqual(sampled.y_trajectory.shape, (6, 7))
            np.testing.assert_array_equal(sampled.joint, repeated.joint)
            np.testing.assert_array_equal(sampled.trajectory[0], sampled.initial_noise)
            np.testing.assert_array_equal(sampled.trajectory[-1], sampled.joint)
            np.testing.assert_array_equal(sampled.times[[0, -1]], [0, 1])
            explicit = sample_joint_model(result.checkpoint_path, 7, steps=5,
                                          initial_noise=sampled.initial_noise)
            np.testing.assert_array_equal(sampled.joint, explicit.joint)
            sampled.save(Path(temporary) / "samples.npz")
            with np.load(Path(temporary) / "samples.npz", allow_pickle=False) as saved:
                self.assertEqual(json.loads(saved["metadata"].item())["method"], "heun")
                np.testing.assert_array_equal(saved["X"], sampled.X)

    def test_invalid_training_contracts_are_rejected(self):
        data = fractional_data()
        with self.assertRaisesRegex(ValueError, "permutation"):
            train_joint_model(data, RectifiedFlowAdapter(3),
                              TrainConfig(mode="permutation_minibatch"))
        with self.assertRaisesRegex(ValueError, "joint_dim"):
            train_joint_model(data, RectifiedFlowAdapter(4), TrainConfig())
        with self.assertRaises(ValueError):
            weighted_objective(torch.ones(3), torch.ones(2))
        with self.assertRaises(FloatingPointError):
            weighted_objective(torch.tensor([float("nan")]), torch.ones(1))
        for config in (TrainConfig(epochs=0), TrainConfig(chunk_size=0),
                       TrainConfig(learning_rate=float("nan")), TrainConfig(seed=-1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()

    def test_heun_and_euler_against_known_time_dependent_velocity(self):
        class Velocity(nn.Module):
            def forward(self, states, times):
                return torch.ones_like(states) * times

        source = torch.tensor([[1., 2.], [-3., 4.]], dtype=torch.float64)
        heun, times, trajectory = integrate_velocity(
            Velocity(), source, steps=10, record_trajectory=True)
        euler, _, no_trajectory = integrate_velocity(Velocity(), source, steps=10, method="euler")
        torch.testing.assert_close(heun, source + 0.5, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(euler, source + 0.45, rtol=1e-12, atol=1e-12)
        self.assertEqual(trajectory.shape, (11, 2, 2))
        self.assertIsNone(no_trajectory)


class CommandLineTests(unittest.TestCase):
    def test_both_solver_flags_train_save_and_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            X = np.array([[0., 1.], [1., 0.], [2., -1.], [-1., 2.]])
            Y = X + np.array([0.5, -0.25])
            np.save(folder / "X.npy", X)
            np.save(folder / "Y.npy", Y)
            for flag, solver in (("--use-permutation", "assignment"),
                                 ("--no-use-permutation", "transportation_lp")):
                with self.subTest(flag=flag), redirect_stdout(io.StringIO()):
                    run = folder / solver
                    main(["train", "--x", str(folder / "X.npy"), "--y", str(folder / "Y.npy"),
                          "--output", str(run), flag, "--epochs", "2", "--hidden-sizes", "8",
                          "--chunk-size", "3"])
                    self.assertTrue((run / "checkpoint.pt").exists())
                    plan = CouplingPlan.load(run / "plan.npz")
                    self.assertEqual(plan.solver, solver)
                    manifest = json.loads((run / "config.json").read_text())
                    self.assertEqual(manifest["coupling"]["solver"], solver)
                    self.assertEqual(manifest["model"]["joint_dim"], 4)
                    np.testing.assert_array_equal(np.load(run / "X.npy"), X)
                    main(["sample", "--checkpoint", str(run / "checkpoint.pt"),
                          "--output", str(run / "samples.npz"), "--count", "6", "--steps", "3",
                          "--record-trajectory"])
                    with np.load(run / "samples.npz", allow_pickle=False) as saved:
                        self.assertEqual(saved["X"].shape, (6, 2))
                        self.assertEqual(saved["Y"].shape, (6, 2))
                        self.assertEqual(saved["trajectory"].shape, (4, 6, 4))


if __name__ == "__main__":
    unittest.main()
