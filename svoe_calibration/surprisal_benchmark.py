"""Compare scalar and vectorized ADEPT and SVoE surprisal."""

import argparse
from bisect import bisect_left
import json
from math import log2
import os
import platform
import socket

import numpy as np
import yaml

from svoe_calibration.benchmark import measure_peak_bytes, measure_time
from svoe_calibration.calibration import prepare_reference


def adept_surprisal_value(particle_scores, weights):
    """Reproduce ADEPT's weighted particle surprisal calculation."""
    likelihood = 0.0

    for weight, score in zip(weights, particle_scores):
        likelihood += weight * np.exp(-score)

    return -np.log(likelihood)


def adept_surprisal_vectorized(particle_scores, weights):
    """Calculate ADEPT surprisal with vectorized NumPy operations."""
    scores = np.asarray(particle_scores, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    return -np.log(np.dot(weight_array, np.exp(-scores)))


def empirical_tail_probability(score, reference):
    """Return an add-one empirical upper-tail probability."""
    first_equal_or_larger = bisect_left(reference, score)
    tail_count = len(reference) - first_equal_or_larger
    return (tail_count + 1) / (len(reference) + 1)


def svoe_surprisal_value(particle_scores, weights, reference):
    """Return empirical Shannon surprisal for one particle ensemble.

    This occupies the same aggregation point as ADEPT's core surprisal:
    each particle discrepancy is mapped to a probability, the particle
    probabilities are combined by the existing weights, and the mixture
    probability is expressed as Shannon information in bits.
    """
    probability = 0.0

    for weight, score in zip(weights, particle_scores):
        probability += weight * empirical_tail_probability(
            score,
            reference,
        )

    return -log2(probability)


def svoe_surprisal_vectorized(particle_scores, weights, reference):
    """Calculate empirical Shannon surprisal with NumPy operations."""
    scores = np.asarray(particle_scores, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)

    first_equal_or_larger = np.searchsorted(
        reference_array,
        scores,
        side="left",
    )
    tail_counts = reference_array.size - first_equal_or_larger
    tail_probabilities = (
        (tail_counts + 1.0) / (reference_array.size + 1.0)
    )
    return -np.log2(np.dot(weight_array, tail_probabilities))


def read_adept_case(path, score_field="all", raw_score_field="raw"):
    """Read saved frame scores and particle-level scores for one case."""
    with open(path) as stream:
        document = json.load(stream)

    if len(document) != 1:
        raise ValueError(
            "{} must contain exactly one case".format(path)
        )

    case_name, result = next(iter(document.items()))

    if score_field not in result:
        raise KeyError(
            "{} does not contain score field {!r}".format(
                path,
                score_field,
            )
        )

    if raw_score_field not in result:
        raise KeyError(
            "{} does not contain raw score field {!r}".format(
                path,
                raw_score_field,
            )
        )

    frame_scores = tuple(float(value) for value in result[score_field])
    particle_scores = tuple(
        tuple(float(value) for value in frame)
        for frame in result[raw_score_field]
    )

    if not frame_scores:
        raise ValueError("{} contains no frame scores".format(path))

    if len(frame_scores) != len(particle_scores):
        raise ValueError(
            "{} has {} frame scores but {} raw-score frames".format(
                path,
                len(frame_scores),
                len(particle_scores),
            )
        )

    if any(not frame for frame in particle_scores):
        raise ValueError(
            "{} contains an empty particle-score frame".format(path)
        )

    return case_name, frame_scores, particle_scores


def equal_weights(count):
    """Return normalized equal weights for a particle population."""
    if count < 1:
        raise ValueError("particle count must be positive")

    weight = 1.0 / count
    return tuple(weight for _ in range(count))


def load_config(path):
    """Load and validate the direct-surprisal benchmark configuration."""
    with open(path) as stream:
        config = yaml.safe_load(stream)

    required = (
        "reference_results",
        "target_results",
        "score_field",
        "raw_score_field",
        "benchmark",
    )
    missing = [key for key in required if key not in config]

    if missing:
        raise KeyError(
            "configuration is missing: {}".format(", ".join(missing))
        )

    benchmark_required = (
        "warmup_passes",
        "iterations_per_repeat",
        "repeats",
        "output_json",
    )
    benchmark_missing = [
        key
        for key in benchmark_required
        if key not in config["benchmark"]
    ]

    if benchmark_missing:
        raise KeyError(
            "benchmark configuration is missing: {}".format(
                ", ".join(benchmark_missing)
            )
        )

    return config


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Directly benchmark ADEPT particle surprisal against a "
            "drop-in empirical SVoE particle surprisal."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the direct-surprisal benchmark YAML configuration.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    score_field = config["score_field"]
    raw_score_field = config["raw_score_field"]
    benchmark_config = config["benchmark"]

    reference_particle_scores = []

    for path in config["reference_results"]:
        _, _, particle_score_frames = read_adept_case(
            path,
            score_field,
            raw_score_field,
        )
        for particle_scores in particle_score_frames:
            reference_particle_scores.extend(particle_scores)

    reference = prepare_reference(reference_particle_scores)
    reference_array = np.asarray(reference, dtype=np.float64)
    adept_inputs = []
    target_cases = []
    particle_counts = set()

    for path in config["target_results"]:
        case_name, _, particle_score_frames = read_adept_case(
            path,
            score_field,
            raw_score_field,
        )
        target_cases.append(case_name)
        for particle_scores in particle_score_frames:
            count = len(particle_scores)
            particle_counts.add(count)
            adept_inputs.append(
                (particle_scores, equal_weights(count))
            )

    frames_per_pass = len(adept_inputs)

    def adept_pass():
        total = 0.0

        for particle_scores, weights in adept_inputs:
            total += adept_surprisal_value(particle_scores, weights)

        return total

    def svoe_pass():
        total = 0.0

        for particle_scores, weights in adept_inputs:
            total += svoe_surprisal_value(
                particle_scores,
                weights,
                reference,
            )

        return total

    def svoe_vectorized_pass():
        total = 0.0

        for particle_scores, weights in adept_inputs:
            total += svoe_surprisal_vectorized(
                particle_scores,
                weights,
                reference_array,
            )

        return total

    def adept_vectorized_pass():
        total = 0.0

        for particle_scores, weights in adept_inputs:
            total += adept_surprisal_vectorized(
                particle_scores,
                weights,
            )

        return total

    for _ in range(benchmark_config["warmup_passes"]):
        adept_pass()
        adept_vectorized_pass()
        svoe_pass()
        svoe_vectorized_pass()

    adept_measurement = measure_time(
        adept_pass,
        benchmark_config["iterations_per_repeat"],
        benchmark_config["repeats"],
        frames_per_pass,
    )
    adept_measurement["peak_incremental_bytes"] = measure_peak_bytes(
        adept_pass
    )

    adept_vectorized_measurement = measure_time(
        adept_vectorized_pass,
        benchmark_config["iterations_per_repeat"],
        benchmark_config["repeats"],
        frames_per_pass,
    )
    adept_vectorized_measurement[
        "peak_incremental_bytes"
    ] = measure_peak_bytes(adept_vectorized_pass)

    svoe_measurement = measure_time(
        svoe_pass,
        benchmark_config["iterations_per_repeat"],
        benchmark_config["repeats"],
        frames_per_pass,
    )
    svoe_measurement["peak_incremental_bytes"] = measure_peak_bytes(
        svoe_pass
    )

    svoe_vectorized_measurement = measure_time(
        svoe_vectorized_pass,
        benchmark_config["iterations_per_repeat"],
        benchmark_config["repeats"],
        frames_per_pass,
    )
    svoe_vectorized_measurement[
        "peak_incremental_bytes"
    ] = measure_peak_bytes(svoe_vectorized_pass)

    released_adept_to_scalar_svoe_ratio = (
        adept_measurement["seconds_per_frame"]
        / svoe_measurement["seconds_per_frame"]
    )
    released_to_vectorized_adept_ratio = (
        adept_measurement["seconds_per_frame"]
        / adept_vectorized_measurement["seconds_per_frame"]
    )
    vectorized_adept_to_scalar_svoe_ratio = (
        adept_vectorized_measurement["seconds_per_frame"]
        / svoe_measurement["seconds_per_frame"]
    )
    scalar_to_vectorized_svoe_ratio = (
        svoe_measurement["seconds_per_frame"]
        / svoe_vectorized_measurement["seconds_per_frame"]
    )
    vectorized_adept_to_vectorized_svoe_ratio = (
        adept_vectorized_measurement["seconds_per_frame"]
        / svoe_vectorized_measurement["seconds_per_frame"]
    )
    released_adept_to_vectorized_svoe_ratio = (
        adept_measurement["seconds_per_frame"]
        / svoe_vectorized_measurement["seconds_per_frame"]
    )
    maximum_adept_absolute_difference = float(
        max(
            abs(
                adept_surprisal_value(particle_scores, weights)
                - adept_surprisal_vectorized(particle_scores, weights)
            )
            for particle_scores, weights in adept_inputs
        )
    )
    maximum_svoe_absolute_difference = float(
        max(
            abs(
                svoe_surprisal_value(
                    particle_scores,
                    weights,
                    reference,
                )
                - svoe_surprisal_vectorized(
                    particle_scores,
                    weights,
                    reference_array,
                )
            )
            for particle_scores, weights in adept_inputs
        )
    )

    output = {
        "runtime": {
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
        },
        "benchmark": {
            "target_cases": target_cases,
            "frames_per_pass": frames_per_pass,
            "reference_particle_score_count": len(reference),
            "particle_counts": sorted(particle_counts),
            "adept_weights": "normalized equal weights per frame",
            "same_particle_inputs_for_both_methods": True,
            "svoe_reference": (
                "flattened particle scores from reference results"
            ),
            "warmup_passes": benchmark_config["warmup_passes"],
            "iterations_per_repeat": benchmark_config[
                "iterations_per_repeat"
            ],
            "repeats": benchmark_config["repeats"],
            "timed_file_io": False,
            "timed_reference_sorting": False,
            "timed_adept_particle_generation": False,
            "timed_svoe_reference_preparation": False,
        },
        "measurements": {
            "adept_released_surprisal": adept_measurement,
            "adept_vectorized_surprisal": (
                adept_vectorized_measurement
            ),
            "svoe_scalar_surprisal": svoe_measurement,
            "svoe_vectorized_surprisal": svoe_vectorized_measurement,
        },
        "comparison": {
            "released_adept_to_scalar_svoe_time_ratio": (
                released_adept_to_scalar_svoe_ratio
            ),
            "released_to_vectorized_adept_time_ratio": (
                released_to_vectorized_adept_ratio
            ),
            "vectorized_adept_to_scalar_svoe_time_ratio": (
                vectorized_adept_to_scalar_svoe_ratio
            ),
            "scalar_to_vectorized_svoe_time_ratio": (
                scalar_to_vectorized_svoe_ratio
            ),
            "vectorized_adept_to_vectorized_svoe_time_ratio": (
                vectorized_adept_to_vectorized_svoe_ratio
            ),
            "released_adept_to_vectorized_svoe_time_ratio": (
                released_adept_to_vectorized_svoe_ratio
            ),
            "maximum_adept_absolute_difference": (
                maximum_adept_absolute_difference
            ),
            "maximum_svoe_absolute_difference_bits": (
                maximum_svoe_absolute_difference
            ),
        },
    }

    output_path = benchmark_config["output_json"]
    output_folder = os.path.dirname(output_path)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    with open(output_path, "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)

    print("host:", output["runtime"]["hostname"])
    print("Python:", output["runtime"]["python_version"])
    print("NumPy:", output["runtime"]["numpy_version"])
    print("frames per pass:", frames_per_pass)
    print("particle counts:", sorted(particle_counts))

    for name, measurement in output["measurements"].items():
        print(name + ":")
        print(
            "  microseconds per frame:",
            measurement["microseconds_per_frame"],
        )
        print(
            "  peak incremental bytes:",
            measurement["peak_incremental_bytes"],
        )
        print(
            "  timing CV percent:",
            measurement["coefficient_of_variation"] * 100,
        )

    print(
        "released ADEPT / vectorized ADEPT time ratio:",
        released_to_vectorized_adept_ratio,
    )
    print(
        "vectorized ADEPT / scalar SVoE time ratio:",
        vectorized_adept_to_scalar_svoe_ratio,
    )
    print(
        "released ADEPT / scalar SVoE time ratio:",
        released_adept_to_scalar_svoe_ratio,
    )
    print(
        "scalar SVoE / vectorized SVoE time ratio:",
        scalar_to_vectorized_svoe_ratio,
    )
    print(
        "vectorized ADEPT / vectorized SVoE time ratio:",
        vectorized_adept_to_vectorized_svoe_ratio,
    )
    print(
        "released ADEPT / vectorized SVoE time ratio:",
        released_adept_to_vectorized_svoe_ratio,
    )
    print(
        "maximum ADEPT absolute difference:",
        maximum_adept_absolute_difference,
    )
    print(
        "maximum SVoE absolute difference (bits):",
        maximum_svoe_absolute_difference,
    )
    print("JSON:", output_path)


if __name__ == "__main__":
    main()
