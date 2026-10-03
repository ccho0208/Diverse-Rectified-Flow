"""Tests for fixed empirical pairs and globally weighted chunks."""

import unittest

import numpy as np
import torch

from diverse_coupling.coupling import CouplingPlan
from diverse_coupling.data import build_paired_dataset


def permutation_plan(permutation):
    permutation = np.asarray(permutation, dtype=np.int64)
    count = len(permutation)
    return CouplingPlan(
        shape=(count, count),
        row_indices=np.arange(count, dtype=np.int64),
        column_indices=permutation,
        masses=np.full(count, 1 / count, dtype=np.float64),
        permutation=permutation,
    )


def fractional_plan():
    return CouplingPlan(
        shape=(2, 2),
        row_indices=np.array([0, 0, 1, 1], dtype=np.int64),
        column_indices=np.array([0, 1, 0, 1], dtype=np.int64),
        masses=np.array([0.1, 0.4, 0.4, 0.1], dtype=np.float64),
    )


class PairedDataTests(unittest.TestCase):
    def test_fractional_plan_matches_dense_asymmetric_loss(self):
        X = np.array([1.0, 3.0])
        Y = np.array([2.0, 7.0])
        dataset = build_paired_dataset(X, Y, fractional_plan())
        batch = dataset.get_batch(np.arange(len(dataset)), dtype=torch.float64)
        x, y = dataset.split_joint(batch.targets)
        loss = (2 * x + 3 * y) ** 2 + x * y
        actual = torch.sum(batch.masses * loss).item()
        dense_loss = (2 * X[:, None] + 3 * Y[None, :]) ** 2 + X[:, None] * Y[None, :]
        expected = np.sum(np.array([[0.1, 0.4], [0.4, 0.1]]) * dense_loss)
        self.assertAlmostEqual(actual, expected)
        self.assertFalse(torch.allclose(batch.masses, torch.full_like(batch.masses, 0.25)))

    def test_chunked_weighted_gradients_equal_full_plan(self):
        dataset = build_paired_dataset(np.array([1.0, 3.0]), np.array([2.0, 7.0]), fractional_plan())
        full_parameter = torch.tensor([0.7, -0.2], dtype=torch.float64, requires_grad=True)
        full_batch = dataset.get_batch(np.arange(len(dataset)), dtype=torch.float64)
        full_loss = torch.sum(full_batch.masses * (full_batch.targets @ full_parameter) ** 2)
        full_loss.backward()

        chunk_parameter = full_parameter.detach().clone().requires_grad_()
        chunk_loss = 0.0
        for indices in dataset.epoch_batches(3):
            batch = dataset.get_batch(indices, dtype=torch.float64)
            contribution = torch.sum(batch.masses * (batch.targets @ chunk_parameter) ** 2)
            contribution.backward()
            chunk_loss += contribution.item()
        self.assertAlmostEqual(chunk_loss, full_loss.item())
        torch.testing.assert_close(chunk_parameter.grad, full_parameter.grad)
        # A chunk's mass is retained, including the single final edge.
        self.assertAlmostEqual(dataset.get_batch([3], dtype=torch.float64).masses.sum().item(), 0.1)

    def test_permutation_preserves_pairs_ids_and_duplicate_atom_labels(self):
        X = np.array([[1.0, 1.0], [1.0, 1.0], [4.0, 5.0]])
        Y = np.array([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]])
        dataset = build_paired_dataset(
            X, Y, permutation_plan([2, 0, 1]),
            x_ids=np.array(["a", "b", "c"]), y_ids=np.array(["u", "v", "w"]),
        )
        self.assertIs(dataset.X, X)
        self.assertIs(dataset.Y, Y)
        batch = dataset.get_batch([1, 0], dtype=torch.float64)
        np.testing.assert_array_equal(batch.x_indices, [1, 0])
        np.testing.assert_array_equal(batch.y_indices, [0, 2])
        np.testing.assert_array_equal(batch.x_ids, ["b", "a"])
        np.testing.assert_array_equal(batch.y_ids, ["u", "w"])
        np.testing.assert_array_equal(batch.targets.numpy(), np.concatenate([X[[1, 0]], Y[[0, 2]]], axis=1))
        self.assertEqual(dataset.joint_dim, 4)

    def test_split_joint_restores_coordinate_and_trajectory_shapes(self):
        dataset = build_paired_dataset(
            np.zeros((2, 2)), np.zeros((2, 2)), permutation_plan([1, 0]),
        )
        states = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
        x, y = dataset.split_joint(states)
        self.assertEqual(tuple(x.shape), (3, 2, 2))
        self.assertEqual(tuple(y.shape), (3, 2, 2))
        torch.testing.assert_close(torch.cat((x, y), dim=-1), states)

        scalar_dataset = build_paired_dataset(np.ones(2), np.ones((2, 2, 2)), permutation_plan([0, 1]))
        x, y = scalar_dataset.split_joint(torch.zeros((3, 2, 5)))
        self.assertEqual(tuple(x.shape), (3, 2))
        self.assertEqual(tuple(y.shape), (3, 2, 2, 2))
        x, y = scalar_dataset.split_joint(torch.zeros(5))
        self.assertEqual(tuple(x.shape), ())
        self.assertEqual(tuple(y.shape), (2, 2))

    def test_epoch_visits_every_edge_once_with_partial_last_batch(self):
        dataset = build_paired_dataset(np.arange(5.0), np.arange(5.0), permutation_plan([4, 3, 2, 1, 0]))
        batches = list(dataset.epoch_batches(2, shuffle=True, rng=np.random.default_rng(19)))
        repeat = list(dataset.epoch_batches(2, shuffle=True, rng=np.random.default_rng(19)))
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(5))
        np.testing.assert_array_equal(np.concatenate(batches), np.concatenate(repeat))
        for indices in batches:
            batch = dataset.get_batch(indices)
            np.testing.assert_array_equal(batch.y_indices, 4 - batch.x_indices)
        np.testing.assert_array_equal(np.concatenate(list(dataset.epoch_batches(2))), np.arange(5))

    def test_rejects_invalid_observations_ids_and_indices(self):
        plan = permutation_plan([0, 1])
        invalid_X = [np.array(1.0), np.array([]), np.ones((2, 0)), np.array([1.0, np.inf]), np.array([1.0, np.nan]), np.array([1j, 2j]), np.array(["a", "b"])]
        for X in invalid_X:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    build_paired_dataset(X, np.ones(2), plan)
        with self.assertRaises(ValueError):
            build_paired_dataset(np.ones(3), np.ones(2), plan)
        for ids in [np.array([1]), np.array([[1], [2]])]:
            with self.assertRaises(ValueError):
                build_paired_dataset(np.ones(2), np.ones(2), plan, x_ids=ids)

        dataset = build_paired_dataset(np.ones(2), np.ones(2), plan)
        for indices in [[0.0], [[0]], [True]]:
            with self.assertRaises(ValueError):
                dataset.get_batch(indices)
        for indices in [[-1], [2], np.array([2**64 - 1], dtype=np.uint64)]:
            with self.assertRaises(IndexError):
                dataset.get_batch(indices)
        with self.assertRaises(ValueError):
            dataset.get_batch([0], dtype=torch.int64)
        with self.assertRaises(ValueError):
            dataset.split_joint(torch.zeros(3))
        for size in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                list(dataset.epoch_batches(size))


if __name__ == "__main__":
    unittest.main()
