"""Checks for offline scientific demo rendering and stored data integrity."""

import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from diverse_coupling.demos.visualize import render_demo


def fixture(directory):
    times = np.linspace(0, 1, 3)
    points = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    noise = np.array([[-0.8, 0.3], [0.8, -0.3]], dtype=float)
    endpoints = noise + [1, 0]
    paths = noise[None] + times[:, None, None] * [1, 0]
    fields = np.broadcast_to(np.array([1, 0]), (3, 4, 2)).copy()
    data = dict(
        times=times, field_points=points, field_x=fields, field_y=-fields,
        bounds=np.array([-2, 2, -2, 2]),
        method_names=np.array(["empirical_optimal", "learned"]),
        output_x=np.stack([endpoints, endpoints]), output_y=np.stack([-endpoints, -endpoints]),
        marginal_path_x=np.stack([paths, paths]), marginal_path_y=np.stack([-paths, -paths]),
        noise_x=np.stack([noise, noise]), noise_y=np.stack([-noise, -noise]),
        component_pair_matrices=np.array([[[0, 0.5], [0.5, 0]], [[0.1, 0.4], [0.4, 0.1]]]),
    )
    np.savez_compressed(Path(directory) / "display.npz", **data)
    report = {"experiment": {"name": "Test experiment"}, "methods": {"learned": {"mismatch_reward": 0.8, "x_sliced_wasserstein": 0.12, "y_sliced_wasserstein": 0.15, "x_off_support_mass": 0.02, "y_off_support_mass": 0.03}}, "round_trip": {"learned": {"x_rmse": 1e-5, "y_rmse": 2e-5}}}
    (Path(directory) / "report.json").write_text(json.dumps(report))
    return data


class DemoVisualizationTests(unittest.TestCase):
    def test_offline_artifacts_contain_controls_and_expected_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = fixture(directory)
            artifacts = render_demo(directory)
            self.assertEqual([path.name for path in artifacts], ["overview.png", "diagnostics.png", "demo.html"])
            for path in artifacts[:2]:
                self.assertGreater(path.stat().st_size, 10000)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            page = artifacts[-1].read_text()
            self.assertIn('id="method"', page)
            self.assertIn('id="time" type="range"', page)
            self.assertIn("inverse of frozen 2D flow", page)
            self.assertIn("independent 4D source", page)
            self.assertIn("SW X", page)
            self.assertIn("Off support Y", page)
            self.assertIn("approximated with random projections", page)
            self.assertNotRegex(page, r'(?:src|href)=["\']https?://')
            match = re.search(r'<script type="application/json" id="demoData">(.*?)</script>', page, re.S)
            payload = json.loads(match.group(1))
            np.testing.assert_array_equal(payload["output_x"], expected["output_x"])
            self.assertEqual(payload["report"]["methods"]["learned"]["mismatch_reward"], 0.8)
            self.assertEqual(payload["report"]["methods"]["learned"]["x_off_support_mass"], 0.02)
            self.assertEqual(payload["labels"]["learned"], "Learned joint flow")

    def test_json_values_and_titles_cannot_close_script_or_insert_markup(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture(directory)
            report_path = Path(directory) / "report.json"
            report = json.loads(report_path.read_text())
            report["description"] = "</script><img src=x onerror=alert(1)>"
            report_path.write_text(json.dumps(report))
            page = render_demo(directory, title="<b>Test & demo</b>")[-1].read_text()
            self.assertNotIn("<img src=x", page)
            self.assertIn("&lt;b&gt;Test &amp; demo&lt;/b&gt;", page)
            match = re.search(r'<script type="application/json" id="demoData">(.*?)</script>', page, re.S)
            self.assertEqual(json.loads(match.group(1))["report"]["description"], report["description"])

    def test_rejects_inconsistent_trajectory_dimensions_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(directory)
            data["marginal_path_x"] = data["marginal_path_x"][:, :-1]
            np.savez_compressed(Path(directory) / "display.npz", **data)
            with self.assertRaisesRegex(ValueError, "marginal_path_x"):
                render_demo(directory)

    def test_rejects_nonfinite_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(directory)
            data["field_x"] = data["field_x"].astype(float)
            data["field_x"][0, 0, 0] = np.nan
            np.savez_compressed(Path(directory) / "display.npz", **data)
            with self.assertRaisesRegex(ValueError, "field_x"):
                render_demo(directory)

    def test_ordinary_transport_figure_keeps_source_identity_and_endpoint_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(directory)
            starts = np.array([[-2, 1], [-2, -1]], dtype=float)
            targets = np.array([[2, -1], [2, 1]], dtype=float)
            linear = (1 - data["times"][:, None, None]) * starts + data["times"][:, None, None] * targets
            ode = linear.copy()
            ode[1, :, 1] = [0.6, -0.6]
            ode[-1] = [[2, 1], [2, -1]]
            data.update(ordinary_linear_path=linear, ordinary_path=ode, reference_x=starts, reference_y=targets)
            np.savez_compressed(Path(directory) / "display.npz", **data)
            artifacts = render_demo(directory)
            self.assertEqual([path.name for path in artifacts], ["overview.png", "diagnostics.png", "ordinary_transport.png", "demo.html"])
            self.assertEqual(artifacts[2].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(artifacts[2].stat().st_size, 10000)
            page = artifacts[-1].read_text()
            payload = json.loads(re.search(r'<script type="application/json" id="demoData">(.*?)</script>', page, re.S).group(1))
            np.testing.assert_array_equal(payload["reference_x"], starts)
            self.assertNotIn("ordinary_path", payload)
            # The two models may induce different endpoints; starts must match.
            np.testing.assert_array_equal(linear[0], ode[0])
            self.assertFalse(np.array_equal(linear[-1], ode[-1]))

    def test_rejects_missing_or_misaligned_ordinary_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(directory)
            paths = data["marginal_path_x"][0]
            data["ordinary_path"] = paths.copy()
            np.savez_compressed(Path(directory) / "display.npz", **data)
            with self.assertRaisesRegex(ValueError, "supplied together"):
                render_demo(directory)
            data["ordinary_linear_path"] = paths.copy()
            data["ordinary_path"][0, 0, 0] += 0.1
            np.savez_compressed(Path(directory) / "display.npz", **data)
            with self.assertRaisesRegex(ValueError, "same starting points"):
                render_demo(directory)


if __name__ == "__main__":
    unittest.main()
