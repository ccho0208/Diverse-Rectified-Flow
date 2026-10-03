"""Distribution geometry and interpretable coupling metrics for the demos."""

from dataclasses import FrozenInstanceError
import json
import unittest

import numpy as np

from diverse_coupling.demos.datasets import ExperimentSpec, build_spec
from diverse_coupling.demos.metrics import (
    component_frequencies,
    component_pair_matrix,
    empirical_independent_matrix,
    evaluate_pairs,
    sliced_wasserstein,
)


class DemoDatasetTests(unittest.TestCase):
    def test_disks_are_uniform_in_area_at_the_agreed_centers(self):
        spec = build_spec("two_disks")
        for side, horizontal in (("source", -2), ("x", 2), ("y", 2)):
            points, labels = spec.sample(100_000, np.random.default_rng(17), side)
            centers = spec.centers(side)
            np.testing.assert_array_equal(centers[:, 0], [horizontal, horizontal])
            np.testing.assert_array_equal(centers[:, 1], [1, -1])
            residuals = points - centers[labels]
            squared_radii = np.sum(residuals**2, axis=1)
            self.assertLessEqual(squared_radii.max(), 0.3**2)
            self.assertAlmostEqual(squared_radii.mean(), 0.3**2 / 2, delta=0.0003)
            np.testing.assert_allclose(np.cov(residuals.T), np.eye(2) * 0.3**2 / 4, atol=0.0003)
            np.testing.assert_array_equal(spec.classify(points, side), labels)
            self.assertTrue(spec.support_mask(points, side).all())

    def test_gaussian_mixture_and_rings_have_correct_geometry(self):
        gmm = build_spec("gmm8")
        np.testing.assert_allclose(np.linalg.norm(gmm.centers(), axis=1), 3.0)
        points, labels = gmm.sample(4096, np.random.default_rng(9))
        np.testing.assert_array_equal(gmm.classify(points), labels)
        for count in (2, 8):
            rings = build_spec("rings", ring_components=count)
            points, labels = rings.sample(20_000, np.random.default_rng(9))
            radius_offsets = np.linalg.norm(points, axis=1) - rings.radii()[labels]
            self.assertTrue(np.all(np.abs(radius_offsets) <= 0.08 + 1e-12))
            self.assertAlmostEqual(float(np.mean(radius_offsets**2)), 0.08**2 / 3, delta=0.00005)
            np.testing.assert_array_equal(rings.classify(points), labels)
            self.assertTrue(rings.support_mask(points).all())
            self.assertFalse(rings.support_mask(np.array([[0.0, 0.0], [2.5, 0.0]])).all())
            self.assertIsNone(rings.centers())

    def test_specs_are_frozen_json_safe_and_reproducible(self):
        for spec in (build_spec("gmm8"), build_spec("two_disks"), build_spec("rings")):
            json.dumps(spec.to_dict(), allow_nan=False)
            x, labels = spec.sample(100, np.random.default_rng(42))
            x2, labels2 = spec.sample(100, np.random.default_rng(42))
            self.assertEqual(x.dtype, np.float64)
            self.assertEqual(labels.dtype, np.int64)
            np.testing.assert_array_equal(x, x2)
            np.testing.assert_array_equal(labels, labels2)
            with self.assertRaises(FrozenInstanceError):
                spec.components = 3

    def test_invalid_specs_counts_points_and_sides_are_rejected(self):
        for name in ("unknown", "two circles"):
            with self.assertRaises(ValueError):
                build_spec(name)
        for count in (1, 3, True, 2.0):
            with self.assertRaises(ValueError):
                build_spec("rings", ring_components=count)
        for args in (("gmm8", 2), ("two_disks", 8)):
            with self.assertRaises(ValueError):
                ExperimentSpec(*args)
        spec = build_spec("two_disks")
        for count in (0, -1, True, 1.2):
            with self.assertRaises(ValueError):
                spec.sample(count, np.random.default_rng())
        for points in (np.empty((0, 2)), np.ones((2, 3)), np.array([[np.nan, 1]]), np.array([["a", "b"]])):
            with self.assertRaises(ValueError):
                spec.classify(points)
        with self.assertRaises(ValueError):
            spec.sample(1, np.random.RandomState(0))
        with self.assertRaises(ValueError):
            spec.centers("z")


class DemoMetricTests(unittest.TestCase):
    def test_component_tables_use_edge_weights_and_correct_side_labels(self):
        spec = build_spec("two_disks")
        x = spec.centers("x")[[0, 0, 1, 1]]
        y = spec.centers("y")[[0, 1, 0, 1]]
        weights = np.array([0.1, 0.4, 0.3, 0.2])
        expected = np.array([[0.1, 0.4], [0.3, 0.2]])
        np.testing.assert_allclose(component_pair_matrix(x, y, spec, weights), expected)
        result = evaluate_pairs(x, y, spec, weights=weights)
        self.assertAlmostEqual(result["mismatch_reward"], 0.7)
        self.assertAlmostEqual(result["squared_distance_reward"], 2.8)
        self.assertEqual(result["x_off_support_mass"], 0)
        self.assertEqual(result["y_off_support_mass"], 0)
        np.testing.assert_allclose(component_frequencies(x, spec, weights=weights), [0.5, 0.5])
        np.testing.assert_allclose(empirical_independent_matrix(x, y, spec), np.full((2, 2), 0.25))
        json.dumps(result, allow_nan=False)

    def test_antipodal_ring_pairs_share_components_but_can_be_far_apart(self):
        spec = build_spec("rings")
        x = np.array([[1.0, 0.0], [3.0, 0.0]])
        same_ring = evaluate_pairs(x, -x, spec)
        other_ring = evaluate_pairs(x, -x[::-1], spec)
        self.assertEqual(same_ring["mismatch_reward"], 0)
        self.assertEqual(other_ring["mismatch_reward"], 1)
        self.assertGreater(same_ring["squared_distance_reward"], other_ring["squared_distance_reward"])
        leakage = evaluate_pairs(np.array([[2.0, 0.0]]), np.array([[0.0, 0.0]]), spec)
        self.assertEqual(leakage["x_off_support_mass"], 1)
        self.assertEqual(leakage["y_off_support_mass"], 1)

    def test_sliced_distance_is_zero_exactly_and_reproducible(self):
        x = np.random.default_rng(7).normal(size=(3100, 2))
        self.assertEqual(sliced_wasserstein(x, x), 0)
        y = x + [2, 0]
        result = sliced_wasserstein(x, y, seed=123)
        self.assertGreater(result, 1)
        self.assertEqual(result, sliced_wasserstein(x, y, seed=123))
        self.assertEqual(sliced_wasserstein(np.zeros((3, 2)), np.zeros((7, 2))), 0)
        self.assertGreater(sliced_wasserstein(np.zeros((3, 2)), np.ones((7, 2))), 0)

    def test_fidelity_reports_marginals_separately(self):
        spec = build_spec("two_disks")
        x, _ = spec.sample(100, np.random.default_rng(1), "x")
        y, _ = spec.sample(100, np.random.default_rng(2), "y")
        result = evaluate_pairs(x, y, spec, reference_X=x, reference_Y=y)
        for side in ("x", "y"):
            self.assertEqual(result[f"{side}_sliced_wasserstein"], 0)
            self.assertEqual(result[f"{side}_mean_error"], 0)
            self.assertEqual(result[f"{side}_covariance_error"], 0)
        weights = np.arange(1, 101)
        result = evaluate_pairs(x, y, spec, reference_X=x, weights=weights)
        self.assertNotIn("x_sliced_wasserstein", result)
        self.assertGreater(result["x_mean_error"], 0)

    def test_invalid_metric_inputs_are_rejected(self):
        spec = build_spec("rings")
        x = np.ones((2, 2))
        for weights in ([0, 0], [-1, 2], [1], [np.inf, 1], ["a", "b"]):
            with self.assertRaises(ValueError):
                component_pair_matrix(x, x, spec, weights)
        with self.assertRaises(ValueError):
            component_pair_matrix(x, x[:1], spec)
        with self.assertRaises(ValueError):
            sliced_wasserstein(x, x, directions=0)


if __name__ == "__main__":
    unittest.main()
