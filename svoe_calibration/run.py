"""Calibrate saved ADEPT frame scores and write SVoE results."""

import argparse
import csv
import json
import math
import os
import time
import tracemalloc
from statistics import mean

import yaml

from svoe_calibration.calibration import calibrate_score, prepare_reference


def read_case(path, score_field):
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
                path, score_field
            )
        )

    scores = [float(score) for score in result[score_field]]

    if not scores:
        raise ValueError("{} contains no frame scores".format(path))

    return case_name, scores


def calibrate_case(scores, reference):
    frames = []

    for frame_index, score in enumerate(scores):
        calibrated = calibrate_score(score, reference)

        frames.append({
            "frame_index": frame_index,
            "adept_score": calibrated["score"],
            "tail_count": calibrated["tail_count"],
            "tail_probability": calibrated["tail_probability"],
            "svoe_bits": calibrated["svoe_bits"],
        })

    bits = [frame["svoe_bits"] for frame in frames]

    return {
        "frame_count": len(frames),
        "adept_score": {
            "minimum": min(scores),
            "mean": mean(scores),
            "maximum": max(scores),
        },
        "svoe_bits": {
            "minimum": min(bits),
            "mean": mean(bits),
            "maximum": max(bits),
        },
        "frames": frames,
    }


def load_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)

    required = (
        "reference_results",
        "target_results",
        "score_field",
        "output_folder",
    )

    missing = [key for key in required if key not in config]

    if missing:
        raise KeyError(
            "configuration is missing: {}".format(", ".join(missing))
        )

    return config


def write_csv(path, cases):
    fieldnames = (
        "case_name",
        "frame_index",
        "adept_score",
        "tail_count",
        "tail_probability",
        "svoe_bits",
    )

    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for case_name, case in cases.items():
            for frame in case["frames"]:
                row = {"case_name": case_name}
                row.update(frame)
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate ADEPT frame scores as SVoE bits."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the SVoE calibration YAML configuration.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    score_field = config["score_field"]

    reference_scores = []
    reference_cases = []

    for path in config["reference_results"]:
        case_name, scores = read_case(path, score_field)
        reference_cases.append(case_name)
        reference_scores.extend(scores)

    reference = prepare_reference(reference_scores)

    targets = {}

    for path in config["target_results"]:
        case_name, scores = read_case(path, score_field)
        targets[case_name] = scores

    start_time = time.perf_counter()
    calibrated_cases = {
        case_name: calibrate_case(scores, reference)
        for case_name, scores in targets.items()
    }
    elapsed_seconds = time.perf_counter() - start_time

    tracemalloc.start()
    for scores in targets.values():
        calibrate_case(scores, reference)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    output = {
        "calibration": {
            "method": "add-one empirical upper-tail calibration",
            "score_field": score_field,
            "reference_cases": reference_cases,
            "reference_count": len(reference),
            "maximum_resolvable_bits": math.log2(len(reference) + 1),
            "elapsed_seconds": elapsed_seconds,
            "peak_bytes": peak_bytes,
        },
        "cases": calibrated_cases,
    }

    output_folder = config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    json_path = os.path.join(
        output_folder,
        "svoe_calibration.json",
    )
    csv_path = os.path.join(
        output_folder,
        "svoe_frame_scores.csv",
    )

    with open(json_path, "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)

    write_csv(csv_path, calibrated_cases)

    print("reference cases:", len(reference_cases))
    print("reference scores:", len(reference))
    print("target cases:", len(calibrated_cases))
    print("calibration seconds:", elapsed_seconds)
    print("peak bytes:", peak_bytes)
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
