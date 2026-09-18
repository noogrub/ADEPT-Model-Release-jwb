"""Reusable analysis functions for ADEPT particle-count experiments."""

import json
import math
import os
from collections import defaultdict
from statistics import mean, stdev

from svoe_calibration.calibration import calibrate_value, prepare_reference


RUN_KEY_FIELDS = ("particle_count", "replicate", "case_name")


def _sample_stdev(values):
    values = list(values)
    return stdev(values) if len(values) > 1 else 0.0


def _percentile(values, percentile):
    """Return a linearly interpolated percentile for finite values."""
    ordered = sorted(float(value) for value in values)

    if not ordered:
        raise ValueError("percentile values cannot be empty")

    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be between 0 and 100")

    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _resolve_path(repository_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(repository_root, path)


def _read_json(path):
    with open(path) as stream:
        return json.load(stream)


def _validate_scores(scores, source):
    values = [float(score) for score in scores]

    if not values:
        raise ValueError("{} contains no frame scores".format(source))

    if not all(math.isfinite(value) for value in values):
        raise ValueError("{} contains a non-finite score".format(source))

    return values


def load_runs(manifest_paths, repository_root=".", score_field="all"):
    """Load and validate run results referenced by experiment manifests."""
    records = []
    seen = set()

    for manifest_path in manifest_paths:
        resolved_manifest = _resolve_path(repository_root, manifest_path)
        manifest = _read_json(resolved_manifest)

        if "runs" not in manifest:
            raise KeyError("{} does not contain runs".format(manifest_path))

        for run in manifest["runs"]:
            missing = [field for field in RUN_KEY_FIELDS if field not in run]
            if missing:
                raise KeyError(
                    "{} run is missing: {}".format(
                        manifest_path,
                        ", ".join(missing),
                    )
                )

            particle_count = int(run["particle_count"])
            replicate = int(run["replicate"])
            case_name = str(run["case_name"])
            key = (particle_count, replicate, case_name)

            if key in seen:
                raise ValueError(
                    "duplicate run particles={} replicate={} case={}".format(
                        particle_count,
                        replicate,
                        case_name,
                    )
                )
            seen.add(key)

            result_path = _resolve_path(repository_root, run["result_path"])
            timing_path = _resolve_path(repository_root, run["timing_path"])
            result_document = _read_json(result_path)
            timing = _read_json(timing_path)

            if case_name not in result_document:
                raise KeyError(
                    "{} does not contain case {!r}".format(
                        result_path,
                        case_name,
                    )
                )

            case_result = result_document[case_name]
            if score_field not in case_result:
                raise KeyError(
                    "{} does not contain score field {!r}".format(
                        result_path,
                        score_field,
                    )
                )

            scores = _validate_scores(case_result[score_field], result_path)
            frame_count = int(timing.get("frame_count", len(scores)))

            if frame_count != len(scores):
                raise ValueError(
                    "{} frame count does not match {}".format(
                        timing_path,
                        result_path,
                    )
                )

            elapsed_seconds = float(timing["elapsed_seconds"])
            if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
                raise ValueError(
                    "{} has invalid elapsed_seconds".format(timing_path)
                )

            records.append({
                "particle_count": particle_count,
                "replicate": replicate,
                "case_name": case_name,
                "scores": scores,
                "frame_count": len(scores),
                "elapsed_seconds": elapsed_seconds,
                "seconds_per_frame": elapsed_seconds / len(scores),
                "execution_mode": timing.get("execution_mode"),
                "result_path": run["result_path"],
                "timing_path": run["timing_path"],
                "manifest_path": manifest_path,
            })

    if not records:
        raise ValueError("no particle experiment runs were loaded")

    return sorted(
        records,
        key=lambda record: (
            record["particle_count"],
            record["replicate"],
            record["case_name"],
        ),
    )


def group_runs(records):
    """Group records by particle count and case name."""
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["particle_count"], record["case_name"])].append(record)

    for group in grouped.values():
        group.sort(key=lambda record: record["replicate"])

    return grouped


def framewise_mean_and_stdev(records, value_field="scores"):
    """Return per-frame mean and sample standard deviation."""
    if not records:
        raise ValueError("records cannot be empty")

    frame_count = len(records[0][value_field])
    if any(len(record[value_field]) != frame_count for record in records):
        raise ValueError("all traces in a group must have equal length")

    frame_means = []
    frame_stdevs = []

    for frame_index in range(frame_count):
        values = [record[value_field][frame_index] for record in records]
        frame_means.append(mean(values))
        frame_stdevs.append(_sample_stdev(values))

    return frame_means, frame_stdevs


def run_summaries(records):
    """Return one scalar summary row per run."""
    rows = []

    for record in records:
        scores = record["scores"]
        row = {
            "particle_count": record["particle_count"],
            "replicate": record["replicate"],
            "case_name": record["case_name"],
            "frame_count": record["frame_count"],
            "adept_minimum": min(scores),
            "adept_mean": mean(scores),
            "adept_maximum": max(scores),
            "elapsed_seconds": record["elapsed_seconds"],
            "seconds_per_frame": record["seconds_per_frame"],
            "execution_mode": record["execution_mode"],
            "result_path": record["result_path"],
            "timing_path": record["timing_path"],
        }

        if "svoe_bits" in record:
            bits = record["svoe_bits"]
            row.update({
                "svoe_minimum_bits": min(bits),
                "svoe_mean_bits": mean(bits),
                "svoe_maximum_bits": max(bits),
                "svoe_reference_count": record["svoe_reference_count"],
            })

        rows.append(row)

    return rows


def aggregate_summaries(records, value_field="scores", value_name="adept"):
    """Summarize run-level and frame-level variation for each group."""
    rows = []

    for (particle_count, case_name), group in sorted(group_runs(records).items()):
        run_means = [mean(record[value_field]) for record in group]
        run_maxima = [max(record[value_field]) for record in group]
        elapsed = [record["elapsed_seconds"] for record in group]
        seconds_per_frame = [record["seconds_per_frame"] for record in group]
        _, frame_stdevs = framewise_mean_and_stdev(group, value_field)

        row = {
            "particle_count": particle_count,
            "case_name": case_name,
            "replicates": len(group),
            "frame_count": len(group[0][value_field]),
            "{}_run_mean_mean".format(value_name): mean(run_means),
            "{}_run_mean_stdev".format(value_name): _sample_stdev(run_means),
            "{}_run_maximum_mean".format(value_name): mean(run_maxima),
            "{}_run_maximum_stdev".format(value_name): _sample_stdev(run_maxima),
            "{}_framewise_stdev_mean".format(value_name): mean(frame_stdevs),
            "{}_framewise_stdev_p95".format(value_name): _percentile(
                frame_stdevs,
                95,
            ),
            "{}_framewise_stdev_maximum".format(value_name): max(frame_stdevs),
            "elapsed_seconds_mean": mean(elapsed),
            "elapsed_seconds_stdev": _sample_stdev(elapsed),
            "seconds_per_frame_mean": mean(seconds_per_frame),
            "seconds_per_frame_stdev": _sample_stdev(seconds_per_frame),
        }
        rows.append(row)

    return rows


def compare_traces(left, right):
    """Return Pearson correlation, MAD, and RMSD for equal-length traces."""
    left = [float(value) for value in left]
    right = [float(value) for value in right]

    if len(left) != len(right):
        raise ValueError("traces must have equal length")
    if not left:
        raise ValueError("traces cannot be empty")

    differences = [a - b for a, b in zip(left, right)]
    mad = mean(abs(value) for value in differences)
    rmsd = math.sqrt(mean(value * value for value in differences))

    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    left_sum_squares = sum((value - left_mean) ** 2 for value in left)
    right_sum_squares = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_sum_squares * right_sum_squares)
    pearson_r = numerator / denominator if denominator else None

    return {
        "pearson_r": pearson_r,
        "mean_absolute_difference": mad,
        "root_mean_square_difference": rmsd,
    }


def compare_group_means(records, baseline_particle_count, value_field="scores"):
    """Compare every group mean trace with the corresponding baseline trace."""
    grouped = group_runs(records)
    baseline = {}

    for (particle_count, case_name), group in grouped.items():
        if particle_count == baseline_particle_count:
            baseline[case_name] = framewise_mean_and_stdev(
                group,
                value_field,
            )[0]

    comparisons = []

    for (particle_count, case_name), group in sorted(grouped.items()):
        if particle_count == baseline_particle_count:
            continue
        if case_name not in baseline:
            raise ValueError(
                "baseline particle count {} has no case {!r}".format(
                    baseline_particle_count,
                    case_name,
                )
            )

        group_mean = framewise_mean_and_stdev(group, value_field)[0]
        comparison = compare_traces(group_mean, baseline[case_name])
        comparison.update({
            "particle_count": particle_count,
            "baseline_particle_count": baseline_particle_count,
            "case_name": case_name,
            "replicates": len(group),
            "baseline_replicates": len(
                grouped[(baseline_particle_count, case_name)]
            ),
        })
        comparisons.append(comparison)

    return comparisons


def compare_runs_to_baseline(records, baseline_particle_count, value_field="scores"):
    """Compare every replicate trace with the baseline mean trace."""
    grouped = group_runs(records)
    baseline = {}

    for (particle_count, case_name), group in grouped.items():
        if particle_count == baseline_particle_count:
            baseline[case_name] = framewise_mean_and_stdev(
                group,
                value_field,
            )[0]

    comparisons = []

    for record in records:
        case_name = record["case_name"]
        if case_name not in baseline:
            raise ValueError(
                "baseline particle count {} has no case {!r}".format(
                    baseline_particle_count,
                    case_name,
                )
            )

        comparison = compare_traces(record[value_field], baseline[case_name])
        comparison.update({
            "particle_count": record["particle_count"],
            "replicate": record["replicate"],
            "case_name": case_name,
            "baseline_particle_count": baseline_particle_count,
        })
        comparisons.append(comparison)

    return comparisons


def leave_one_replicate_out_svoe(records, reference_case_names):
    """Calibrate each run without using controls from its own replicate."""
    reference_case_names = set(reference_case_names)
    if not reference_case_names:
        raise ValueError("reference_case_names cannot be empty")

    by_particle_count = defaultdict(list)
    for record in records:
        by_particle_count[record["particle_count"]].append(record)

    calibrated_records = []

    for particle_count, particle_records in sorted(by_particle_count.items()):
        available_cases = {
            record["case_name"]
            for record in particle_records
            if record["case_name"] in reference_case_names
        }
        missing_cases = reference_case_names - available_cases
        if missing_cases:
            raise ValueError(
                "particle count {} is missing reference cases: {}".format(
                    particle_count,
                    ", ".join(sorted(missing_cases)),
                )
            )

        for target in particle_records:
            reference_scores = []
            for candidate in particle_records:
                if candidate["replicate"] == target["replicate"]:
                    continue
                if candidate["case_name"] not in reference_case_names:
                    continue
                reference_scores.extend(candidate["scores"])

            if not reference_scores:
                raise ValueError(
                    "particle count {} needs at least two replicates".format(
                        particle_count
                    )
                )

            reference = prepare_reference(reference_scores)
            bits = []
            tail_probabilities = []

            for score in target["scores"]:
                _, tail_probability, svoe_bits = calibrate_value(
                    score,
                    reference,
                )
                tail_probabilities.append(tail_probability)
                bits.append(svoe_bits)

            calibrated = dict(target)
            calibrated["svoe_bits"] = bits
            calibrated["tail_probabilities"] = tail_probabilities
            calibrated["svoe_reference_count"] = len(reference)
            calibrated_records.append(calibrated)

    return sorted(
        calibrated_records,
        key=lambda record: (
            record["particle_count"],
            record["replicate"],
            record["case_name"],
        ),
    )


def efficiency_metrics(aggregate_rows, baseline_particle_count):
    """Add speedup and runtime reduction relative to a baseline."""
    baseline_seconds = {
        row["case_name"]: row["elapsed_seconds_mean"]
        for row in aggregate_rows
        if row["particle_count"] == baseline_particle_count
    }

    output = []
    for row in aggregate_rows:
        case_name = row["case_name"]
        if case_name not in baseline_seconds:
            raise ValueError(
                "baseline particle count {} has no case {!r}".format(
                    baseline_particle_count,
                    case_name,
                )
            )

        enriched = dict(row)
        baseline = baseline_seconds[case_name]
        elapsed = row["elapsed_seconds_mean"]
        enriched["baseline_particle_count"] = baseline_particle_count
        enriched["speedup_vs_baseline"] = baseline / elapsed
        enriched["runtime_reduction_percent"] = (1.0 - elapsed / baseline) * 100.0
        output.append(enriched)

    return output


def top_frames(records, count=5):
    """Return the highest-scoring frames from every run."""
    if count < 1:
        raise ValueError("count must be positive")

    rows = []
    for record in records:
        ranked = sorted(
            enumerate(record["scores"]),
            key=lambda item: (-item[1], item[0]),
        )[:count]

        for rank, (frame_index, score) in enumerate(ranked, 1):
            rows.append({
                "particle_count": record["particle_count"],
                "replicate": record["replicate"],
                "case_name": record["case_name"],
                "rank": rank,
                "frame_index": frame_index,
                "adept_score": score,
            })

    return rows


def inspect_result_frames(path, frame_indices, case_name=None):
    """Extract aggregate, raw-particle, and location data for selected frames."""
    document = _read_json(path)

    if case_name is None:
        if len(document) != 1:
            raise ValueError("case_name is required for multi-case results")
        case_name = next(iter(document))

    if case_name not in document:
        raise KeyError("{} does not contain case {!r}".format(path, case_name))

    result = document[case_name]
    scores = result["all"]
    raw = result.get("raw")
    locations = result.get("location")
    rows = []

    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= len(scores):
            raise IndexError("frame {} is outside the result".format(frame_index))

        row = {
            "frame_index": frame_index,
            "adept_score": scores[frame_index],
        }

        if raw is not None:
            row["raw_particle_scores"] = raw[frame_index]
            row["raw_particle_count"] = len(raw[frame_index])

        if locations is not None and len(locations) >= 2:
            row["location_x"] = locations[0][frame_index]
            row["location_y"] = locations[1][frame_index]
            row["location_x_count"] = len(locations[0][frame_index])
            row["location_y_count"] = len(locations[1][frame_index])

        rows.append(row)

    return {
        "case_name": case_name,
        "result_path": path,
        "frames": rows,
    }
