import os
import unittest

from svoe_calibration.particle_experiment import (
    DirectExecutionGate,
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


if __name__ == "__main__":
    unittest.main()

