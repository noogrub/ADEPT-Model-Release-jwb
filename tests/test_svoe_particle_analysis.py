import json
import math
import os
import tempfile
import unittest

from svoe_calibration.particle_analysis import (
    aggregate_summaries,
    compare_traces,
    efficiency_metrics,
    framewise_mean_and_stdev,
    inspect_result_frames,
    leave_one_replicate_out_svoe,
    load_runs,
    top_frames,
)
from svoe_calibration.analyze_particles import (
    plot_baseline_comparisons,
    plot_efficiency,
    plot_replicates,
)


class ParticleAnalysisTests(unittest.TestCase):

    def record(self, particle_count, replicate, case_name, scores, elapsed=2.0):
        return {
            "particle_count": particle_count,
            "replicate": replicate,
            "case_name": case_name,
            "scores": list(scores),
            "frame_count": len(scores),
            "elapsed_seconds": elapsed,
            "seconds_per_frame": elapsed / len(scores),
            "execution_mode": "adept_pool",
            "result_path": "result.json",
            "timing_path": "timing.json",
            "manifest_path": "manifest.json",
        }

    def test_framewise_mean_and_sample_stdev(self):
        records = [
            self.record(2, 1, "case", [1.0, 4.0]),
            self.record(2, 2, "case", [3.0, 8.0]),
        ]

        means, deviations = framewise_mean_and_stdev(records)

        self.assertEqual(means, [2.0, 6.0])
        self.assertAlmostEqual(deviations[0], math.sqrt(2.0))
        self.assertAlmostEqual(deviations[1], math.sqrt(8.0))

    def test_trace_comparison(self):
        comparison = compare_traces([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])

        self.assertGreater(comparison["pearson_r"], 0.98)
        self.assertAlmostEqual(
            comparison["mean_absolute_difference"],
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            comparison["root_mean_square_difference"],
            math.sqrt(1.0 / 3.0),
        )

    def test_leave_one_replicate_out_reference(self):
        records = [
            self.record(4, 1, "case_a", [1.0, 2.0]),
            self.record(4, 1, "case_b", [3.0, 4.0]),
            self.record(4, 2, "case_a", [2.0, 3.0]),
            self.record(4, 2, "case_b", [4.0, 5.0]),
        ]

        calibrated = leave_one_replicate_out_svoe(
            records,
            ["case_a", "case_b"],
        )

        self.assertEqual(len(calibrated), 4)
        self.assertTrue(
            all(record["svoe_reference_count"] == 4 for record in calibrated)
        )

        target = next(
            record for record in calibrated
            if record["replicate"] == 1 and record["case_name"] == "case_b"
        )
        self.assertEqual(len(target["svoe_bits"]), 2)
        self.assertGreater(target["svoe_bits"][1], target["svoe_bits"][0])

    def test_aggregate_and_efficiency_metrics(self):
        records = [
            self.record(2, 1, "case", [1.0, 2.0], elapsed=2.0),
            self.record(2, 2, "case", [1.0, 4.0], elapsed=4.0),
            self.record(8, 1, "case", [1.0, 2.0], elapsed=8.0),
            self.record(8, 2, "case", [1.0, 2.0], elapsed=8.0),
        ]

        aggregate = aggregate_summaries(records)
        enriched = efficiency_metrics(aggregate, baseline_particle_count=8)
        particle_two = next(
            row for row in enriched if row["particle_count"] == 2
        )

        self.assertAlmostEqual(particle_two["elapsed_seconds_mean"], 3.0)
        self.assertAlmostEqual(particle_two["speedup_vs_baseline"], 8.0 / 3.0)
        self.assertAlmostEqual(
            particle_two["runtime_reduction_percent"],
            62.5,
        )

    def test_top_frames_are_ranked(self):
        records = [self.record(3, 1, "case", [2.0, 5.0, 1.0, 4.0])]

        rows = top_frames(records, count=2)

        self.assertEqual([row["frame_index"] for row in rows], [1, 3])
        self.assertEqual([row["rank"] for row in rows], [1, 2])

    def test_load_manifest_and_inspect_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            run_folder = os.path.join(folder, "run")
            os.makedirs(run_folder)
            result_relative = os.path.join("run", "result.json")
            timing_relative = os.path.join("run", "timing.json")
            manifest_relative = "manifest.json"

            result = {
                "case_a": {
                    "all": [0.5, 2.5],
                    "raw": [[0.4, 0.6], [2.0, 3.0]],
                    "location": [
                        [[1.0, 2.0], [3.0]],
                        [[4.0, 5.0], [6.0]],
                    ],
                }
            }
            timing = {
                "frame_count": 2,
                "elapsed_seconds": 1.5,
                "execution_mode": "adept_pool",
            }
            manifest = {
                "runs": [{
                    "particle_count": 2,
                    "replicate": 1,
                    "case_name": "case_a",
                    "result_path": result_relative,
                    "timing_path": timing_relative,
                }]
            }

            for relative, document in (
                    (result_relative, result),
                    (timing_relative, timing),
                    (manifest_relative, manifest),
            ):
                with open(os.path.join(folder, relative), "w") as stream:
                    json.dump(document, stream)

            records = load_runs(
                [manifest_relative],
                repository_root=folder,
            )
            inspected = inspect_result_frames(
                os.path.join(folder, result_relative),
                [1],
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["scores"], [0.5, 2.5])
            self.assertEqual(inspected["frames"][0]["raw_particle_count"], 2)
            self.assertEqual(inspected["frames"][0]["location_x_count"], 1)

    def test_plots_are_written(self):
        records = [
            self.record(2, 1, "case", [0.0, 1.0, 2.0], elapsed=2.0),
            self.record(2, 2, "case", [0.0, 1.5, 2.5], elapsed=3.0),
            self.record(8, 1, "case", [0.0, 1.0, 2.0], elapsed=8.0),
            self.record(8, 2, "case", [0.0, 1.0, 2.0], elapsed=8.0),
        ]
        plot_config = {
            "figure_width": 4,
            "figure_height": 3,
            "dpi": 50,
            "line_width": 0.8,
            "band_alpha": 0.16,
        }

        with tempfile.TemporaryDirectory() as folder:
            paths = plot_replicates(
                records,
                folder,
                plot_config,
                "scores",
                "ADEPT score",
                "adept",
            )
            paths.extend(plot_baseline_comparisons(
                records,
                8,
                folder,
                plot_config,
                "scores",
                "ADEPT score",
                "adept",
            ))
            aggregate = aggregate_summaries(records)
            paths.append(plot_efficiency(aggregate, folder, plot_config))

            self.assertEqual(len(paths), 4)
            self.assertTrue(all(os.path.exists(path) for path in paths))


if __name__ == "__main__":
    unittest.main()
