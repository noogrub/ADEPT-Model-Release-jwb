"""Create reproducible reports for ADEPT particle-count experiments."""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml

from svoe_calibration.particle_analysis import (
    aggregate_summaries,
    compare_group_means,
    compare_runs_to_baseline,
    efficiency_metrics,
    framewise_mean_and_stdev,
    group_runs,
    leave_one_replicate_out_svoe,
    load_runs,
    run_summaries,
    top_frames,
)


DEFAULT_PLOT_CONFIG = {
    "figure_width": 16,
    "figure_height": 8,
    "dpi": 150,
    "line_width": 0.8,
    "band_alpha": 0.16,
}


def load_analysis_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)

    required = (
        "manifests",
        "reference_case_names",
        "baseline_particle_count",
        "output_folder",
    )
    missing = [key for key in required if key not in config]

    if missing:
        raise KeyError(
            "configuration is missing: {}".format(", ".join(missing))
        )
    if not config["manifests"]:
        raise ValueError("manifests cannot be empty")
    if not config["reference_case_names"]:
        raise ValueError("reference_case_names cannot be empty")

    validated = dict(config)
    validated["repository_root"] = config.get("repository_root", ".")
    validated["score_field"] = config.get("score_field", "all")
    validated["top_frames_per_run"] = int(
        config.get("top_frames_per_run", 5)
    )
    validated["baseline_particle_count"] = int(
        config["baseline_particle_count"]
    )

    plot_config = dict(DEFAULT_PLOT_CONFIG)
    plot_config.update(config.get("plot", {}))
    validated["plot"] = plot_config

    return validated


def write_json(path, document):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)


def write_csv(path, rows):
    rows = list(rows)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(path, "w", newline="") as stream:
        if not fieldnames:
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_ylim(bottom=0)
    axis.margins(x=0)
    axis.tick_params(direction="out")


def _case_title(case_name):
    return case_name.replace("_", " ")


def _save_figure(figure, path, dpi):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_replicates(records, output_folder, plot_config, value_field, y_label, suffix):
    paths = []
    figure_folder = os.path.join(output_folder, "figures", "replicates")

    for (particle_count, case_name), group in sorted(group_runs(records).items()):
        figure, axis = plt.subplots(
            figsize=(
                plot_config["figure_width"],
                plot_config["figure_height"],
            )
        )

        for record in group:
            axis.plot(
                range(len(record[value_field])),
                record[value_field],
                linewidth=plot_config["line_width"],
                label="Replicate {:03d}".format(record["replicate"]),
            )

        axis.set_title(
            "{}: N = {}".format(_case_title(case_name), particle_count)
        )
        axis.set_xlabel("Frame")
        axis.set_ylabel(y_label)
        axis.set_xlim(0, len(group[0][value_field]) - 1)
        axis.legend(frameon=False)
        style_axis(axis)
        figure.tight_layout()

        path = os.path.join(
            figure_folder,
            "{}_n{:03d}_replicates_{}.png".format(
                case_name,
                particle_count,
                suffix,
            ),
        )
        _save_figure(figure, path, plot_config["dpi"])
        paths.append(path)

    return paths


def plot_baseline_comparisons(
        records,
        baseline_particle_count,
        output_folder,
        plot_config,
        value_field,
        y_label,
        suffix):
    paths = []
    grouped = group_runs(records)
    figure_folder = os.path.join(output_folder, "figures", "comparisons")

    baseline_groups = {
        case_name: group
        for (particle_count, case_name), group in grouped.items()
        if particle_count == baseline_particle_count
    }

    for (particle_count, case_name), group in sorted(grouped.items()):
        if particle_count == baseline_particle_count:
            continue

        baseline_group = baseline_groups[case_name]
        target_mean, target_sd = framewise_mean_and_stdev(group, value_field)
        baseline_mean, baseline_sd = framewise_mean_and_stdev(
            baseline_group,
            value_field,
        )
        frames = list(range(len(target_mean)))

        figure, axis = plt.subplots(
            figsize=(
                plot_config["figure_width"],
                plot_config["figure_height"],
            )
        )

        series = (
            (
                particle_count,
                target_mean,
                target_sd,
                "#4C78A8",
            ),
            (
                baseline_particle_count,
                baseline_mean,
                baseline_sd,
                "#F58518",
            ),
        )

        for count, values, deviations, color in series:
            lower = [max(0.0, value - deviation) for value, deviation in zip(values, deviations)]
            upper = [value + deviation for value, deviation in zip(values, deviations)]
            axis.plot(
                frames,
                values,
                color=color,
                linewidth=plot_config["line_width"],
                label="N = {} mean".format(count),
            )
            axis.fill_between(
                frames,
                lower,
                upper,
                color=color,
                alpha=plot_config["band_alpha"],
                linewidth=0,
            )

        axis.set_title(_case_title(case_name))
        axis.set_xlabel("Frame")
        axis.set_ylabel(y_label)
        axis.set_xlim(0, len(frames) - 1)
        axis.legend(frameon=False)
        style_axis(axis)
        figure.tight_layout()

        path = os.path.join(
            figure_folder,
            "{}_n{:03d}_n{:03d}_{}.png".format(
                case_name,
                particle_count,
                baseline_particle_count,
                suffix,
            ),
        )
        _save_figure(figure, path, plot_config["dpi"])
        paths.append(path)

    return paths


def plot_efficiency(aggregate_rows, output_folder, plot_config):
    by_case = defaultdict(list)
    for row in aggregate_rows:
        by_case[row["case_name"]].append(row)

    figure, axis = plt.subplots(
        figsize=(
            plot_config["figure_width"],
            plot_config["figure_height"],
        )
    )

    for case_name, rows in sorted(by_case.items()):
        rows.sort(key=lambda row: row["particle_count"])
        particle_counts = [row["particle_count"] for row in rows]
        elapsed = [row["elapsed_seconds_mean"] for row in rows]
        elapsed_sd = [row["elapsed_seconds_stdev"] for row in rows]
        axis.errorbar(
            particle_counts,
            elapsed,
            yerr=elapsed_sd,
            marker="o",
            markersize=4,
            capsize=3,
            linewidth=plot_config["line_width"],
            label=_case_title(case_name),
        )

    axis.set_title("ADEPT particle-count runtime")
    axis.set_xlabel("Particles")
    axis.set_ylabel("Mean physics time per video (seconds)")
    axis.set_xlim(left=0)
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()

    path = os.path.join(
        output_folder,
        "figures",
        "particle_runtime.png",
    )
    _save_figure(figure, path, plot_config["dpi"])
    return path


def svoe_frame_rows(records):
    rows = []

    for record in records:
        for frame_index, (score, probability, bits) in enumerate(zip(
                record["scores"],
                record["tail_probabilities"],
                record["svoe_bits"],
        )):
            rows.append({
                "particle_count": record["particle_count"],
                "replicate": record["replicate"],
                "case_name": record["case_name"],
                "frame_index": frame_index,
                "adept_score": score,
                "tail_probability": probability,
                "svoe_bits": bits,
                "reference_count": record["svoe_reference_count"],
            })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ADEPT particle-count experiments."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the particle-analysis YAML configuration.",
    )
    args = parser.parse_args()

    config = load_analysis_config(args.config)
    records = load_runs(
        config["manifests"],
        repository_root=config["repository_root"],
        score_field=config["score_field"],
    )
    particle_counts = sorted({record["particle_count"] for record in records})
    baseline_particle_count = config["baseline_particle_count"]

    if baseline_particle_count not in particle_counts:
        raise ValueError(
            "baseline particle count {} was not loaded".format(
                baseline_particle_count
            )
        )

    calibrated = leave_one_replicate_out_svoe(
        records,
        config["reference_case_names"],
    )

    run_rows = run_summaries(calibrated)
    adept_aggregate = efficiency_metrics(
        aggregate_summaries(records, "scores", "adept"),
        baseline_particle_count,
    )
    svoe_aggregate = aggregate_summaries(
        calibrated,
        "svoe_bits",
        "svoe_bits",
    )
    adept_group_comparison = compare_group_means(
        records,
        baseline_particle_count,
        "scores",
    )
    svoe_group_comparison = compare_group_means(
        calibrated,
        baseline_particle_count,
        "svoe_bits",
    )
    adept_run_comparison = compare_runs_to_baseline(
        records,
        baseline_particle_count,
        "scores",
    )

    output_folder = config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    paths = {
        "run_summary_csv": os.path.join(output_folder, "run_summary.csv"),
        "aggregate_summary_csv": os.path.join(
            output_folder,
            "aggregate_summary.csv",
        ),
        "svoe_aggregate_summary_csv": os.path.join(
            output_folder,
            "svoe_aggregate_summary.csv",
        ),
        "adept_group_comparison_csv": os.path.join(
            output_folder,
            "adept_group_comparison.csv",
        ),
        "svoe_group_comparison_csv": os.path.join(
            output_folder,
            "svoe_group_comparison.csv",
        ),
        "adept_run_comparison_csv": os.path.join(
            output_folder,
            "adept_run_comparison.csv",
        ),
        "top_frames_csv": os.path.join(output_folder, "top_frames.csv"),
        "svoe_frames_csv": os.path.join(output_folder, "svoe_frames.csv"),
        "analysis_json": os.path.join(output_folder, "analysis.json"),
    }

    write_csv(paths["run_summary_csv"], run_rows)
    write_csv(paths["aggregate_summary_csv"], adept_aggregate)
    write_csv(paths["svoe_aggregate_summary_csv"], svoe_aggregate)
    write_csv(paths["adept_group_comparison_csv"], adept_group_comparison)
    write_csv(paths["svoe_group_comparison_csv"], svoe_group_comparison)
    write_csv(paths["adept_run_comparison_csv"], adept_run_comparison)
    write_csv(
        paths["top_frames_csv"],
        top_frames(records, config["top_frames_per_run"]),
    )
    write_csv(paths["svoe_frames_csv"], svoe_frame_rows(calibrated))

    figure_paths = []
    figure_paths.extend(plot_replicates(
        records,
        output_folder,
        config["plot"],
        "scores",
        "ADEPT score",
        "adept",
    ))
    figure_paths.extend(plot_replicates(
        calibrated,
        output_folder,
        config["plot"],
        "svoe_bits",
        "SVoE (bits)",
        "svoe",
    ))
    figure_paths.extend(plot_baseline_comparisons(
        records,
        baseline_particle_count,
        output_folder,
        config["plot"],
        "scores",
        "ADEPT score",
        "adept",
    ))
    figure_paths.extend(plot_baseline_comparisons(
        calibrated,
        baseline_particle_count,
        output_folder,
        config["plot"],
        "svoe_bits",
        "SVoE (bits)",
        "svoe",
    ))
    figure_paths.append(
        plot_efficiency(adept_aggregate, output_folder, config["plot"])
    )

    reference_counts = sorted({
        record["svoe_reference_count"] for record in calibrated
    })
    report = {
        "analysis": {
            "method": "leave-one-replicate-out empirical upper-tail calibration",
            "score_field": config["score_field"],
            "reference_case_names": config["reference_case_names"],
            "baseline_particle_count": baseline_particle_count,
            "particle_counts": particle_counts,
            "run_count": len(records),
            "reference_counts_per_fold": reference_counts,
            "shaded_bands": "mean plus or minus one sample standard deviation",
        },
        "run_summary": run_rows,
        "adept_aggregate_summary": adept_aggregate,
        "svoe_aggregate_summary": svoe_aggregate,
        "adept_group_comparison": adept_group_comparison,
        "svoe_group_comparison": svoe_group_comparison,
        "adept_run_comparison": adept_run_comparison,
        "figure_paths": figure_paths,
    }
    write_json(paths["analysis_json"], report)

    print("runs:", len(records))
    print("particle counts:", particle_counts)
    print("reference counts per fold:", reference_counts)
    for name, path in paths.items():
        print("{}: {}".format(name, path))
    print("figures:", len(figure_paths))


if __name__ == "__main__":
    main()
