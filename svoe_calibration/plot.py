"""Plot ADEPT frame scores and calibrated SVoE bits."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_ylim(bottom=0)
    axis.margins(x=0)
    axis.tick_params(direction="out")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ADEPT and SVoE frame scores."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the SVoE calibration YAML configuration.",
    )
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    plot_config = config["plot"]

    with open(plot_config["input_json"]) as stream:
        results = json.load(stream)

    output_folder = plot_config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    for case_name, case in results["cases"].items():
        frames = case["frames"]

        frame_indices = [
            frame["frame_index"] for frame in frames
        ]
        adept_scores = [
            frame["adept_score"] for frame in frames
        ]
        svoe_bits = [
            frame["svoe_bits"] for frame in frames
        ]

        figure, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(
                plot_config["figure_width"],
                plot_config["figure_height"],
            ),
            constrained_layout=True,
        )

        axes[0].plot(
            frame_indices,
            adept_scores,
            color="#4C78A8",
            linewidth=plot_config["line_width"],
        )
        axes[0].set_ylabel("ADEPT score")
        style_axis(axes[0])

        axes[1].plot(
            frame_indices,
            svoe_bits,
            color="#E45756",
            linewidth=plot_config["line_width"],
        )
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("SVoE (bits)")
        style_axis(axes[1])

        figure.suptitle(case_name.replace("_", " "))

        output_path = os.path.join(
            output_folder,
            "{}_adept_svoe.png".format(case_name),
        )

        figure.savefig(
            output_path,
            dpi=plot_config["dpi"],
            bbox_inches="tight",
        )
        plt.close(figure)

        print(output_path)


if __name__ == "__main__":
    main()
