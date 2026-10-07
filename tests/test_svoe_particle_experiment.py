import os
import json
import tempfile
import unittest

from svoe_calibration.particle_experiment import (
    DirectExecutionGate,
    completed_run,
    run_directory,
    validate_experiment_config,
)


class ParticleExperimentTests(unittest.TestCase):

    def base_config(self):
        return {
            "physics_config": "experiments/default_physics.yaml",
            "case_names": ["case_a"],
            "particle_counts": [1],
            "replicates": 2,
            "output_root": "output/example",
            "overwrite_existing": False,
            "execution": {
                "direct_single_particle": True,
            },
        }

    def test_valid_config(self):
        config = validate_experiment_config(self.base_config())

        self.assertEqual(config["particle_counts"], [1])
        self.assertEqual(config["replicates"], 2)

    def test_particle_count_must_be_positive(self):
        config = self.base_config()
        config["particle_counts"] = [0]

        with self.assertRaises(ValueError):
            validate_experiment_config(config)

    def test_run_directory_is_unique(self):
        path = run_directory(
            "output/example",
            particle_count=1,
            replicate=2,
            case_name="case_a",
        )

        self.assertEqual(
            path,
            os.path.join(
                "output/example",
                "particles_001",
                "replicate_002",
                "case_a",
            ),
        )

    def test_direct_gate_selects_direct_path(self):
        gate = DirectExecutionGate()

        self.assertFalse(gate.acquire(False))

    def test_completed_run_returns_validated_metadata(self):
        with tempfile.TemporaryDirectory() as output_root:
            config = self.base_config()
            config["output_root"] = output_root
            output_directory = run_directory(
                output_root,
                particle_count=1,
                replicate=1,
                case_name="case_a",
            )
            os.makedirs(output_directory)

            result_path = os.path.join(output_directory, "result.json")
            timing_path = os.path.join(output_directory, "timing.json")

            with open(result_path, "w") as stream:
                json.dump({"case_a": {"all": [0.1, 0.2]}}, stream)

            with open(timing_path, "w") as stream:
                json.dump(
                    {
                        "case_name": "case_a",
                        "particle_count": 1,
                        "replicate": 1,
                        "execution_mode": "direct",
                        "frame_count": 2,
                        "elapsed_seconds": 4.0,
                        "seconds_per_frame": 2.0,
                    },
                    stream,
                )

            run = completed_run(config, 1, 1, "case_a")

            self.assertEqual(run["frame_count"], 2)
            self.assertEqual(run["elapsed_seconds"], 4.0)
            self.assertEqual(run["result_path"], result_path)

    def test_completed_run_returns_none_when_output_is_absent(self):
        with tempfile.TemporaryDirectory() as output_root:
            config = self.base_config()
            config["output_root"] = output_root

            self.assertIsNone(completed_run(config, 1, 1, "case_a"))

    def test_completed_run_rejects_incomplete_output_pair(self):
        with tempfile.TemporaryDirectory() as output_root:
            config = self.base_config()
            config["output_root"] = output_root
            output_directory = run_directory(
                output_root,
                particle_count=1,
                replicate=1,
                case_name="case_a",
            )
            os.makedirs(output_directory)

            with open(os.path.join(output_directory, "result.json"), "w") as stream:
                json.dump({"case_a": {"all": [0.1]}}, stream)

            with self.assertRaises(RuntimeError):
                completed_run(config, 1, 1, "case_a")


if __name__ == "__main__":
    unittest.main()
