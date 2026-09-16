"""Benchmark the marginal computational cost of SVoE calibration."""

import argparse
import gc
import json
import os
import platform
import socket
import statistics
import time
import tracemalloc

import yaml

from svoe_calibration.calibration import (
    calibrate_value,
    prepare_reference,
)
from svoe_calibration.run import calibrate_case, read_case


def measure_time(function, iterations, repeats, frames_per_pass):
    repeat_seconds = []

    for _ in range(repeats):
        gc.collect()
        gc.disable()

        try:
            start = time.perf_counter()

            for _ in range(iterations):
                function()

            elapsed = time.perf_counter() - start
        finally:
            gc.enable()

        repeat_seconds.append(elapsed)

    minimum_repeat_seconds = min(repeat_seconds)
    maximum_repeat_seconds = max(repeat_seconds)
    mean_repeat_seconds = statistics.mean(repeat_seconds)
    median_repeat_seconds = statistics.median(repeat_seconds)

    if len(repeat_seconds) > 1:
        stdev_repeat_seconds = statistics.stdev(repeat_seconds)
    else:
        stdev_repeat_seconds = 0.0

    coefficient_of_variation = (
        stdev_repeat_seconds / mean_repeat_seconds
        if mean_repeat_seconds
        else 0.0
    )

    seconds_per_pass = median_repeat_seconds / iterations
    seconds_per_frame = seconds_per_pass / frames_per_pass

    return {
        "repeat_seconds": repeat_seconds,
        "minimum_repeat_seconds": minimum_repeat_seconds,
        "median_repeat_seconds": median_repeat_seconds,
        "maximum_repeat_seconds": maximum_repeat_seconds,
        "mean_repeat_seconds": mean_repeat_seconds,
        "stdev_repeat_seconds": stdev_repeat_seconds,
        "coefficient_of_variation": coefficient_of_variation,
        "seconds_per_pass": seconds_per_pass,
        "seconds_per_frame": seconds_per_frame,
        "microseconds_per_frame": seconds_per_frame * 1e6,
        "frames_per_second": 1.0 / seconds_per_frame,
    }


def measure_peak_bytes(function):
    gc.collect()
    tracemalloc.start()

    result = function()
    _, peak_bytes = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    # Keep the result alive until after peak memory is read.
    del result

    return peak_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SVoE calibration overhead."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the SVoE calibration YAML configuration.",
    )
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    score_field = config["score_field"]
    benchmark_config = config["benchmark"]

    reference_scores = []

    for path in config["reference_results"]:
        _, scores = read_case(path, score_field)
        reference_scores.extend(scores)

    reference = prepare_reference(reference_scores)

    targets = {}

    for path in config["target_results"]:
        case_name, scores = read_case(path, score_field)
        targets[case_name] = scores

    frames_per_pass = sum(
        len(scores) for scores in targets.values()
    )

    def calculation_only_pass():
        for scores in targets.values():
            for score in scores:
                calibrate_value(score, reference)

    def integrated_pass():
        return {
            case_name: calibrate_case(scores, reference)
            for case_name, scores in targets.items()
        }

    for _ in range(benchmark_config["warmup_passes"]):
        calculation_only_pass()
        integrated_pass()

    calculation_only = measure_time(
        calculation_only_pass,
        benchmark_config["iterations_per_repeat"],
        benchmark_config["repeats"],
        frames_per_pass,
    )
    calculation_only["peak_incremental_bytes"] = measure_peak_bytes(
        calculation_only_pass
    )

    integrated = measure_time(
        integrated_pass,
        benchmark_config["iterations_per_repeat"],
        benchmark_config["repeats"],
        frames_per_pass,
    )
    integrated["peak_incremental_bytes"] = measure_peak_bytes(
        integrated_pass
    )

    output = {
        "runtime": {
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "benchmark": {
            "reference_count": len(reference),
            "target_cases": len(targets),
            "frames_per_pass": frames_per_pass,
            "warmup_passes": benchmark_config["warmup_passes"],
            "iterations_per_repeat": benchmark_config[
                "iterations_per_repeat"
            ],
            "repeats": benchmark_config["repeats"],
            "timed_file_io": False,
            "timed_reference_sorting": False,
        },
        "measurements": {
            "calculation_only": calculation_only,
            "integrated_output": integrated,
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
    print("frames per pass:", frames_per_pass)

    for name, measurement in output["measurements"].items():
        print(name + ":")
        print(
            "  microseconds per frame:",
            measurement["microseconds_per_frame"],
        )
        print(
            "  frames per second:",
            measurement["frames_per_second"],
        )
        print(
            "  peak incremental bytes:",
            measurement["peak_incremental_bytes"],
        )
        print(
            "  timing CV percent:",
            measurement["coefficient_of_variation"] * 100,
        )

    print("JSON:", output_path)


if __name__ == "__main__":
    main()
