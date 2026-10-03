"""Numerical and empirical-objective checks for the shared demo RF tools."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from diverse_coupling.demos.flows import (
    FlowTrainingConfig, integrate, load_flow, round_trip, train_flow, velocity_grid,
)
from diverse_coupling.models import RectifiedFlowAdapter
from diverse_coupling.demos.sources import SourceDistribution


class DemoFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bidirectional_time_field_and_batching(self):
        class TimeVelocity(nn.Module):
            def forward(self, states, times):
                if (times < 0).any() or (times > 1).any():
                    raise AssertionError("inverse integration must stay in [0, 1]")
                return torch.ones_like(states) * times[:, None]

        model = TimeVelocity().train()
        source = np.array([[1., 2.], [-3., 4.], [5., -6.]])
        forward = integrate(model, source, steps=7, batch_size=2, record_trajectory=True)
        np.testing.assert_allclose(forward.endpoints, source + .5, rtol=0, atol=1e-14)
        self.assertTrue(model.training)
        self.assertEqual(forward.trajectory.shape, (8, 3, 2))
        np.testing.assert_array_equal(forward.trajectory[0], source)
        np.testing.assert_array_equal(forward.trajectory[-1], forward.endpoints)
        reverse = integrate(model, forward.endpoints, steps=7, start_time=1, end_time=0,
                            batch_size=1, record_trajectory=True)
        np.testing.assert_allclose(reverse.endpoints, source, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(reverse.times[[0, -1]], [1, 0])
        euler = integrate(model, source, steps=10, method="euler")
        np.testing.assert_allclose(euler.endpoints, source + .45, rtol=0, atol=1e-14)

    def test_reverse_reconstruction_error_decreases_with_resolution(self):
        class ExpVelocity(nn.Module):
            def forward(self, states, times):
                return states

        points = np.array([[1., 2.], [-3., .5]])
        coarse = round_trip(ExpVelocity(), points, steps=10, record_trajectory=True)
        fine = round_trip(ExpVelocity(), points, steps=40)
        np.testing.assert_allclose(fine["noise"], points / np.e, rtol=2e-4, atol=0)
        self.assertLess(fine["max_error"], coarse["max_error"] / 20)
        self.assertEqual(coarse["reverse_trajectory"].shape, (11, 2, 2))
        self.assertAlmostEqual(fine["rmse"], np.sqrt(np.mean(fine["errors"]**2)))

    def test_velocity_grid_reports_actual_time_conditioned_field(self):
        class Field(nn.Module):
            def forward(self, states, times):
                return states + torch.stack((times, -2 * times), dim=1)

        model = Field().train()
        result = velocity_grid(model, (-2, 2, -1, 3), [0, .5, 1], grid_size=3)
        self.assertTrue(model.training)
        self.assertEqual(result["points"].shape, (9, 2))
        self.assertEqual(result["velocities"].shape, (3, 9, 2))
        for index, time in enumerate(result["times"]):
            np.testing.assert_allclose(result["velocities"][index],
                                       result["points"] + [time, -2 * time])

    def test_fixed_source_fractional_objective_matches_explicit_adam_step(self):
        targets = np.array([[1., 2.], [3., -4.], [-2., 0.], [.5, -1.]])
        sources = targets + np.array([[-1., 0.], [1., 2.], [0., -1.], [-2., 1.]])
        masses = np.array([.1, .2, .3, .4])
        config = FlowTrainingConfig(epochs=1, seed=5, dtype="float64", chunk_size=3,
                                    checkpoint_every=1)
        reference = RectifiedFlowAdapter(2, (5,), seed=5).double()
        generator = torch.Generator().manual_seed(5)
        t = torch.rand(4, dtype=torch.float64, generator=generator)
        loss = (reference.per_example_loss(torch.tensor(targets), noise=torch.tensor(sources),
                                          times=t) * torch.tensor(masses)).sum()
        optimizer = torch.optim.Adam(reference.parameters(), lr=config.learning_rate)
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as temporary:
            model = train_flow(targets, config, output_dir=temporary, source_points=sources,
                               masses=masses, hidden_sizes=(5,))
            _, payload = load_flow(Path(temporary) / "checkpoint.pt")
            self.assertAlmostEqual(payload["history"][0]["loss"], loss.item(), places=12)
            self.assertEqual(payload["source_kind"], "fixed")
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_gaussian_rng_and_full_plan_updates_are_chunk_equivalent(self):
        targets = np.arange(28, dtype=float).reshape(7, 4) / 9
        masses = np.arange(1, 8, dtype=float)
        masses /= masses.sum()
        config = FlowTrainingConfig(epochs=4, seed=9, dtype="float64", chunk_size=7,
                                    checkpoint_every=2)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            full = train_flow(targets, config, output_dir=folder / "full", masses=masses,
                              hidden_sizes=(8, 8))
            chunked = train_flow(targets, replace(config, chunk_size=2),
                                 output_dir=folder / "chunks", masses=masses, hidden_sizes=(8, 8))
            _, full_payload = load_flow(folder / "full/checkpoint.pt")
            _, chunked_payload = load_flow(folder / "chunks/checkpoint.pt")
            np.testing.assert_allclose([r["loss"] for r in full_payload["history"]],
                                       [r["loss"] for r in chunked_payload["history"]],
                                       rtol=1e-12, atol=1e-12)
            for a, b in zip(full.parameters(), chunked.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_disk_source_full_weighted_objective_matches_explicit_step(self):
        source = SourceDistribution("uniform_disks", ((-2., 1.), (-2., -1.)), copies=2)
        targets = np.array([[2., 1., 2., -1.], [2., -1., 2., 1.], [2., 1., 2., 1.]])
        masses = np.array([.2, .3, .5])
        config = FlowTrainingConfig(epochs=1, seed=7, dtype="float64", chunk_size=2)
        generator = torch.Generator().manual_seed(7)
        noise = source.sample_torch(3, 4, generator=generator, device=torch.device("cpu"), dtype=torch.float64)
        times = torch.rand(3, generator=generator, dtype=torch.float64)
        reference = RectifiedFlowAdapter(4, (5,), seed=7).double()
        loss = (reference.per_example_loss(torch.tensor(targets), noise=noise, times=times) * torch.tensor(masses)).sum()
        optimizer = torch.optim.Adam(reference.parameters(), lr=config.learning_rate)
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as temporary:
            trained = train_flow(targets, config, output_dir=temporary, masses=masses,
                                 source_distribution=source, hidden_sizes=(5,))
            for actual, expected in zip(trained.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            with self.assertRaisesRegex(ValueError, "data identity"):
                train_flow(targets, config, output_dir=temporary, masses=masses,
                           source_distribution=SourceDistribution(), hidden_sizes=(5,), resume=True)

    def test_disk_source_area_density_product_law_and_antithetic_reflection(self):
        source = SourceDistribution("uniform_disks", ((-2., 1.), (-2., -1.)), copies=2)
        generator = torch.Generator().manual_seed(31)
        points = source.sample_torch(50_000, 4, generator=generator,
                                    device=torch.device("cpu"), dtype=torch.float64).numpy()
        blocks = points.reshape(-1, 2)
        centers = np.column_stack((np.full(len(blocks), -2.), np.where(blocks[:, 1] > 0, 1., -1.)))
        squared = np.sum((blocks - centers)**2, axis=1)
        self.assertLessEqual(squared.max(), .3**2)
        self.assertAlmostEqual(squared.mean(), .3**2 / 2, delta=.0003)
        self.assertAlmostEqual(np.mean((points[:, 1] > 0) != (points[:, 3] > 0)), .5, delta=.01)
        reflected = source.antithetic(points)
        np.testing.assert_allclose(source.antithetic(reflected), points, atol=1e-15)
        np.testing.assert_allclose(reflected, [-4., 0., -4., 0.] - points)

    def test_resume_matches_uninterrupted_training_and_rejects_changed_data(self):
        targets = np.array([[1., 2.], [3., -4.], [-2., 0.]])
        config = FlowTrainingConfig(epochs=6, seed=12, dtype="float64", chunk_size=2,
                                    checkpoint_every=2)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            full = train_flow(targets, config, output_dir=folder / "full", hidden_sizes=(7,))
            train_flow(targets, replace(config, epochs=3), output_dir=folder / "resume",
                       hidden_sizes=(7,))
            records = []
            resumed = train_flow(targets, config, output_dir=folder / "resume", hidden_sizes=(7,),
                                 resume=True, progress=records.append)
            self.assertEqual([row["epoch"] for row in records], [4, 5, 6])
            _, full_payload = load_flow(folder / "full/checkpoint.pt")
            _, resumed_payload = load_flow(folder / "resume/checkpoint.pt")
            self.assertEqual(full_payload["history"], resumed_payload["history"])
            for a, b in zip(full.parameters(), resumed.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "data identity"):
                train_flow(targets + 1, config, output_dir=folder / "resume", hidden_sizes=(7,),
                           resume=True)
            with self.assertRaisesRegex(ValueError, "learning_rate"):
                train_flow(targets, replace(config, learning_rate=.02),
                           output_dir=folder / "resume", hidden_sizes=(7,), resume=True)

    def test_scheduler_stop_saves_complete_epoch_and_can_resume(self):
        config = FlowTrainingConfig(epochs=3, seed=2, checkpoint_every=50)
        targets = np.array([[1., 0.], [-1., 0.]])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(InterruptedError, "epoch 1"):
                train_flow(targets, config, output_dir=temporary, hidden_sizes=(4,),
                           stop_requested=lambda: True)
            _, payload = load_flow(Path(temporary) / "checkpoint.pt")
            self.assertEqual(payload["completed_epochs"], 1)
            train_flow(targets, config, output_dir=temporary, hidden_sizes=(4,), resume=True)
            _, payload = load_flow(Path(temporary) / "checkpoint.pt")
            self.assertEqual(payload["completed_epochs"], 3)
            self.assertEqual([r["epoch"] for r in payload["history"]], [1, 2, 3])

    def test_validation_and_dimension_mismatch(self):
        for config in (FlowTrainingConfig(epochs=0), FlowTrainingConfig(chunk_size=0),
                       FlowTrainingConfig(dtype="float16"), FlowTrainingConfig(seed=-1),
                       FlowTrainingConfig(checkpoint_every=0)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
        with tempfile.TemporaryDirectory() as temporary:
            config = FlowTrainingConfig(epochs=1)
            with self.assertRaisesRegex(ValueError, "sum to one"):
                train_flow(np.ones((2, 2)), config, output_dir=temporary, masses=np.ones(2))
            with self.assertRaisesRegex(ValueError, "same shape"):
                train_flow(np.ones((2, 2)), config, output_dir=temporary,
                           source_points=np.ones((2, 1)))
        model = RectifiedFlowAdapter(4, (4,), seed=1)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            integrate(model, np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "2D"):
            velocity_grid(model, (-1, 1, -1, 1), [0, 1])


if __name__ == "__main__":
    unittest.main()
