"""Small mathematical checks for rewards and empirical optimal plans."""

from itertools import permutations
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diverse_coupling.costs import build_reward_matrix
from diverse_coupling.coupling import (
    CouplingPlan,
    coupling_from_permutation,
    solve_assignment,
    solve_optimal_coupling,
)


class AssignmentTests(unittest.TestCase):
    def test_global_optimum_is_not_independent_row_maximization(self):
        C = np.array([[10.0, 9.0], [9.0, 0.0]])
        plan = solve_optimal_coupling(C)
        np.testing.assert_array_equal(plan.permutation, [1, 0])
        np.testing.assert_array_equal(plan.to_dense(), [[0.0, 0.5], [0.5, 0.0]])
        self.assertEqual(plan.reward, 9.0)

    def test_assignment_and_lp_agree_with_exhaustive_optima(self):
        rng = np.random.default_rng(45)
        matrices = [
            np.array([[2.0, 7.0, 3.0], [6.0, 4.0, 1.0], [5.0, 2.0, 8.0]]),
            -rng.uniform(1.0, 20.0, size=(4, 4)),
            rng.normal(size=(5, 5)),
            np.zeros((3, 3)),
            np.full((3, 3), -7.0),
            np.array([[1.0, 0.0], [0.0, 2.0]]) * 1e-200,
        ]
        for C in matrices:
            with self.subTest(C=C):
                n = len(C)
                optimum = max(
                    sum(C[i, sigma[i]] for i in range(n)) / n
                    for sigma in permutations(range(n))
                )
                assignment = solve_optimal_coupling(C)
                reference = solve_optimal_coupling(C, use_permutation=False)
                for plan in (assignment, reference):
                    np.testing.assert_allclose(plan.reward, optimum, rtol=1e-10, atol=0)
                    dense = plan.to_dense()
                    np.testing.assert_allclose(dense.sum(axis=0), 1 / n, atol=1e-10)
                    np.testing.assert_allclose(dense.sum(axis=1), 1 / n, atol=1e-10)
                    self.assertTrue(plan.validate()["valid"])
                self.assertIsNone(reference.permutation)

    def test_invalid_rewards_rejected_by_both_backends(self):
        bad_matrices = [
            np.empty((0, 0)), np.ones((2, 3)), np.ones(3),
            np.array([[np.nan]]), np.array([[np.inf]]),
            np.array([[1j]]), np.array([["1"]]),
        ]
        for matrix in bad_matrices:
            for flag in (True, False):
                with self.subTest(matrix=matrix, flag=flag):
                    with self.assertRaises(ValueError):
                        solve_optimal_coupling(matrix, use_permutation=flag)
        with self.assertRaises(ValueError):
            solve_optimal_coupling(np.eye(2), use_permutation="False")

    def test_lp_is_invariant_to_large_row_reward_offsets(self):
        residual = np.array([[71.0, 32.0], [23.0, 98.0]])
        for offsets in (np.full((2, 1), 1e15), np.array([[1e15], [-1e15]])):
            C = residual + offsets
            plan = solve_optimal_coupling(C, use_permutation=False)
            np.testing.assert_array_equal(plan.to_dense(), np.eye(2) / 2)
            self.assertEqual(float(np.sum(plan.to_dense() * residual)), 84.5)

    def test_lp_handles_finite_rewards_with_extreme_opposite_signs(self):
        C = np.array([[1e308, -1e308], [-1e308, 1e308]])
        plan = solve_optimal_coupling(C, use_permutation=False)
        np.testing.assert_array_equal(plan.to_dense(), np.eye(2) / 2)
        self.assertEqual(plan.reward, 1e308)

    def test_permutation_atoms_remain_distinct_under_ties(self):
        sigma = solve_assignment(np.zeros((8, 8)))
        np.testing.assert_array_equal(np.sort(sigma), np.arange(8))
        plan = coupling_from_permutation(sigma)
        self.assertEqual(len(plan.masses), 8)
        self.assertEqual(plan.row_indices.dtype, np.int64)
        self.assertEqual(plan.masses.dtype, np.float64)
        for sigma in ([], [0, 0], [1, 2], [-1, 0], [0.0, 1.0], [[0, 1]]):
            with self.subTest(sigma=sigma):
                with self.assertRaises(ValueError):
                    coupling_from_permutation(np.asarray(sigma))


class PlanTests(unittest.TestCase):
    def fractional_plan(self, **changes):
        kwargs = dict(
            shape=(2, 2),
            row_indices=np.array([0, 0, 1, 1]),
            column_indices=np.array([0, 1, 0, 1]),
            masses=np.array([0.1, 0.4, 0.4, 0.1]),
            solver="test_fractional",
            reward=4.5,
            diagnostics={"note": "fixed weighted plan", "iterations": 0},
        )
        kwargs.update(changes)
        return CouplingPlan(**kwargs)

    def test_fractional_probability_plan_and_sparse_conversion(self):
        plan = self.fractional_plan()
        self.assertEqual(plan.validate()["total_mass"], 1.0)
        np.testing.assert_array_equal(plan.to_dense(), [[0.1, 0.4], [0.4, 0.1]])
        self.assertEqual(plan.to_sparse().format, "csr")
        self.assertIsNone(plan.permutation)

    def test_plan_save_reload_has_no_pickled_arrays(self):
        plans = [self.fractional_plan(), solve_optimal_coupling(np.eye(3))]
        with tempfile.TemporaryDirectory() as folder:
            for i, plan in enumerate(plans):
                path = Path(folder) / str(i) / "plan.npz"
                plan.save(path)
                restored = CouplingPlan.load(path)
                with np.load(path, allow_pickle=False) as archive:
                    for name in archive.files:
                        self.assertNotEqual(archive[name].dtype.kind, "O")
                np.testing.assert_array_equal(restored.to_dense(), plan.to_dense())
                self.assertEqual(restored.solver, plan.solver)
                self.assertEqual(restored.reward, plan.reward)
                self.assertEqual(restored.diagnostics, plan.diagnostics)
                if plan.permutation is not None:
                    np.testing.assert_array_equal(restored.permutation, plan.permutation)

    def test_invalid_plan_entries_and_marginals(self):
        invalid = [
            {"shape": (2, 3)},
            {"masses": np.array([0.1, 0.4, -0.4, 0.9])},
            {"masses": np.array([0.0, 0.5, 0.5, 0.0])},
            {"masses": np.array([np.nan, 0.4, 0.4, 0.1])},
            {"masses": np.array([0.1, 0.4, 0.4])},
            {"masses": np.array([0.2, 0.3, 0.4, 0.1])},
            {"column_indices": np.array([0, 0, 0, 1])},
            {"row_indices": np.array([0, 0, 2, 2])},
            {"row_indices": np.array([0, 0, -1, 1])},
            {"reward": float("inf")},
            {"permutation": np.array([1, 0])},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.fractional_plan(**changes).validate()
        with self.assertRaises(ValueError):
            self.fractional_plan(row_indices=np.array([0.0, 0.0, 1.0, 1.0]))
        for tolerance in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                self.fractional_plan().validate(atol=tolerance)

    def test_total_mass_alone_is_not_sufficient(self):
        plan = CouplingPlan(
            shape=(2, 2),
            row_indices=np.array([0, 1]),
            column_indices=np.array([0, 0]),
            masses=np.array([0.5, 0.5]),
        )
        with self.assertRaises(ValueError):
            plan.validate()


class RewardTests(unittest.TestCase):
    def test_squared_separation_matches_independent_pair_calculation(self):
        x = np.array([[1.0, 2.0], [-3.0, 1.0], [2.0, 4.0]])
        y = np.array([[2.0, 1.0], [5.0, -1.0], [0.0, 0.0]])
        expected = np.array([[np.sum((a - b) ** 2) for b in y] for a in x])
        for block_size in (None, 1, 2, 20):
            result = build_reward_matrix(x, y, block_size=block_size)
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(
            build_reward_matrix(x.reshape(3, 1, 2), y), expected
        )

    def test_component_reward_uses_labels_not_distance(self):
        x = np.zeros((3, 2))
        y = np.ones((3, 4))
        result = build_reward_matrix(
            x, y, "component_mismatch",
            x_labels=np.array(["a", "b", "a"]),
            y_labels=np.array(["b", "b", "a"]), block_size=1,
        )
        np.testing.assert_array_equal(result, [[1, 1, 0], [0, 0, 1], [1, 1, 0]])

    def test_callable_keeps_tensor_observation_shapes(self):
        x = np.arange(12.0).reshape(3, 2, 2)
        y = np.arange(6.0).reshape(3, 1, 2)
        calls = []

        def reward(a, b):
            calls.append((a.shape, b.shape))
            return 2 * a.sum(axis=(1, 2))[:, None] - b.sum(axis=(1, 2))[None, :]

        actual = build_reward_matrix(x, y, reward, block_size=2)
        expected = 2 * x.sum(axis=(1, 2))[:, None] - y.sum(axis=(1, 2))[None, :]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(calls, [((2, 2, 2), (3, 1, 2)), ((1, 2, 2), (3, 1, 2))])

    def test_invalid_data_reward_labels_and_overflow_rejected(self):
        x = np.zeros((2, 2))
        cases = [
            (np.zeros((0, 2)), np.zeros((0, 2)), {}),
            (x, np.zeros((3, 2)), {}),
            (np.full((2, 2), np.nan), x, {}),
            (np.array([["1"], ["2"]]), x, {}),
            (x, np.zeros((2, 3)), {}),
            (np.array([[1e308], [-1e308]]), np.zeros((2, 1)), {}),
            (x, x, {"block_size": 0}),
            (x, x, {"block_size": 1.5}),
            (x, x, {"block_size": True}),
            (x, x, {"reward": "unknown"}),
            (x, x, {"reward": "component_mismatch"}),
            (x, x, {"reward": "component_mismatch", "x_labels": [0], "y_labels": [0, 1]}),
            (x, x, {"reward": "component_mismatch", "x_labels": [0, np.inf], "y_labels": [0, 1]}),
            (x, x, {"reward": lambda a, b: np.ones(len(a))}),
            (x, x, {"reward": lambda a, b: np.full((len(a), len(b)), np.inf)}),
            (x, x, {"reward": lambda a, b: np.full((len(a), len(b)), 1j)}),
        ]
        for a, b, kwargs in cases:
            with self.subTest(kwargs=kwargs, x_shape=a.shape, y_shape=b.shape):
                with self.assertRaises(ValueError):
                    build_reward_matrix(a, b, **kwargs)


if __name__ == "__main__":
    unittest.main()
