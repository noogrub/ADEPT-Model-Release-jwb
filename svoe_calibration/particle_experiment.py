"""Run isolated ADEPT particle-count experiments without plotting.

This module imports ADEPT's existing physics and particle-filter code but does
not modify it. Each particle-count/replicate/case combination receives a
separate output directory.
"""

import argparse
import json
import os
import time
from multiprocessing import Semaphore

import yaml


REQUIRED_CONFIG_KEYS = (
    "physics_config",
    "case_names",
    "particle_counts",
    "replicates",
    "output_root",
    "overwrite_existing",
    "execution",
)


class DirectExecutionGate:
    """Force FilterUpdater to use its direct, non-pool execution path."""

    def acquire(self, blocking=False):
        return False

    def release(self):
        raise RuntimeError("direct execution gate cannot be released")


def validate_experiment_config(config):
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]

    if missing:
        raise KeyError(
            "configuration is missing: {}".format(", ".join(missing))
        )

    if not config["case_names"]:
        raise ValueError("case_names cannot be empty")

    particle_counts = [int(value) for value in config["particle_counts"]]

    if not particle_counts:
        raise ValueError("particle_counts cannot be empty")

    if any(value < 1 for value in particle_counts):
        raise ValueError("particle counts must be positive")

    replicates = int(config["replicates"])

    if replicates < 1:
        raise ValueError("replicates must be positive")

    execution = config["execution"]

    if "direct_single_particle" not in execution:
        raise KeyError("execution.direct_single_particle is required")

    validated = dict(config)
    validated["particle_counts"] = particle_counts
    validated["replicates"] = replicates

    return validated


def load_experiment_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)

    return validate_experiment_config(config)


def run_directory(output_root, particle_count, replicate, case_name):
    return os.path.join(
        output_root,
        "particles_{:03d}".format(particle_count),
        "replicate_{:03d}".format(replicate),
        case_name,
    )


def completed_run(experiment, particle_count, replicate, case_name):
    """Return validated metadata for a completed run, or None."""
    output_directory = run_directory(
        experiment["output_root"],
        particle_count,
        replicate,
        case_name,
    )
    result_path = os.path.join(output_directory, "result.json")
    timing_path = os.path.join(output_directory, "timing.json")
    result_exists = os.path.exists(result_path)
    timing_exists = os.path.exists(timing_path)

    if not result_exists and not timing_exists:
        return None

    if result_exists != timing_exists:
        raise RuntimeError(
            "incomplete run output: expected both {} and {}".format(
                result_path,
                timing_path,
            )
        )

    with open(result_path) as stream:
        result_document = json.load(stream)

    if case_name not in result_document:
        raise ValueError(
            "completed result is missing case {!r}: {}".format(
                case_name,
                result_path,
            )
        )

    scores = result_document[case_name].get("all")

    if not isinstance(scores, list):
        raise ValueError(
            "completed result is missing frame scores: {}".format(
                result_path,
            )
        )

    with open(timing_path) as stream:
        timing = json.load(stream)

    expected = {
        "case_name": case_name,
        "particle_count": particle_count,
        "replicate": replicate,
        "frame_count": len(scores),
    }

    for key, value in expected.items():
        if timing.get(key) != value:
            raise ValueError(
                "completed timing metadata mismatch for {} in {}".format(
                    key,
                    timing_path,
                )
            )

    return {
        "case_name": case_name,
        "particle_count": particle_count,
        "replicate": replicate,
        "execution_mode": timing["execution_mode"],
        "frame_count": timing["frame_count"],
        "elapsed_seconds": timing["elapsed_seconds"],
        "seconds_per_frame": timing["seconds_per_frame"],
        "result_path": result_path,
        "timing_path": timing_path,
    }


def build_physics_config(base_config_path, particle_count, case_name):
    from tools.physics_defaults import _C as default_config

    config = default_config.clone()
    config.merge_from_file(base_config_path)
    config.defrost()
    config.MODEL.N_PARTICLES = particle_count
    config.CASE_NAMES = [case_name]
    config.freeze()

    return config


def read_case_inputs(config, case_name):
    from utils.io import read_serialized

    annotation_path = os.path.join(
        config.ANNOTATION_FOLDER,
        case_name,
        case_name + "_ann.yaml",
    )
    observation_path = os.path.join(
        config.OBSERVATION_FOLDER,
        case_name + ".json",
    )

    annotation = read_serialized(annotation_path)
    observation_document = read_serialized(observation_path)
    scenes = observation_document["scene"]

    for scene in scenes:
        for obj in scene["objects"]:
            if "color" not in obj:
                obj["color"] = "green"

    return annotation["camera"], scenes


def execution_gate(particle_count, direct_single_particle):
    if particle_count == 1 and direct_single_particle:
        return DirectExecutionGate(), "direct"

    return Semaphore(1), "adept_pool"


def write_json(path, document):
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)


def run_case(experiment, particle_count, replicate, case_name):
    from models.particle_filter import FilterUpdater
    from utils.io import write_serialized

    physics_config = build_physics_config(
        experiment["physics_config"],
        particle_count,
        case_name,
    )
    camera, scenes = read_case_inputs(physics_config, case_name)

    output_directory = run_directory(
        experiment["output_root"],
        particle_count,
        replicate,
        case_name,
    )
    result_path = os.path.join(output_directory, "result.json")
    timing_path = os.path.join(output_directory, "timing.json")

    if os.path.exists(result_path) and not experiment["overwrite_existing"]:
        raise FileExistsError(
            "result already exists: {}".format(result_path)
        )

    os.makedirs(output_directory, exist_ok=True)

    gate, execution_mode = execution_gate(
        particle_count,
        experiment["execution"]["direct_single_particle"],
    )

    updater = FilterUpdater(
        physics_config,
        scenes[0]["objects"],
        case_name,
        camera,
        gate,
    )

    start_time = time.perf_counter()
    updater.run(scenes[1:])
    elapsed_seconds = time.perf_counter() - start_time

    score = updater.get_score()
    frame_count = len(score["all"])

    write_serialized({case_name: score}, result_path)
    write_json(
        timing_path,
        {
            "case_name": case_name,
            "particle_count": particle_count,
            "replicate": replicate,
            "execution_mode": execution_mode,
            "frame_count": frame_count,
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_frame": elapsed_seconds / frame_count,
            "timed_input_loading": False,
            "timed_output_writing": False,
            "timed_plotting": False,
        },
    )

    return {
        "case_name": case_name,
        "particle_count": particle_count,
        "replicate": replicate,
        "execution_mode": execution_mode,
        "frame_count": frame_count,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_frame": elapsed_seconds / frame_count,
        "result_path": result_path,
        "timing_path": timing_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run isolated ADEPT particle-count experiments."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the particle experiment YAML configuration.",
    )
    args = parser.parse_args()

    experiment = load_experiment_config(args.config)
    runs = []

    for particle_count in experiment["particle_counts"]:
        for replicate in range(1, experiment["replicates"] + 1):
            for case_name in experiment["case_names"]:
                if not experiment["overwrite_existing"]:
                    previous = completed_run(
                        experiment,
                        particle_count,
                        replicate,
                        case_name,
                    )
                    if previous is not None:
                        runs.append(previous)
                        print(
                            "skipping completed particles={} replicate={} "
                            "case={}".format(
                                particle_count,
                                replicate,
                                case_name,
                            )
                        )
                        continue

                print(
                    "running particles={} replicate={} case={}".format(
                        particle_count,
                        replicate,
                        case_name,
                    )
                )
                result = run_case(
                    experiment,
                    particle_count,
                    replicate,
                    case_name,
                )
                runs.append(result)
                print(
                    "completed in {:.6f} seconds".format(
                        result["elapsed_seconds"]
                    )
                )

    manifest_path = os.path.join(
        experiment["output_root"],
        "manifest.json",
    )
    write_json(
        manifest_path,
        {
            "experiment_config": experiment,
            "runs": runs,
        },
    )

    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
