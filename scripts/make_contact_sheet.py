"""Create a labeled contact sheet from ADEPT frame images."""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def grid_shape(image_count, columns):
    if image_count < 1:
        raise ValueError("image_count must be positive")
    if columns < 1:
        raise ValueError("columns must be positive")
    columns = min(columns, image_count)
    rows = int(math.ceil(image_count / float(columns)))
    return rows, columns


def main():
    parser = argparse.ArgumentParser(
        description="Create a contact sheet from ADEPT frame images."
    )
    parser.add_argument("--case", required=True, help="ADEPT case name.")
    parser.add_argument(
        "--frames",
        required=True,
        nargs="+",
        type=int,
        help="Zero-based frame indices to include.",
    )
    parser.add_argument(
        "--input-folder",
        default="output/default_physics/imgs",
        help="Folder containing <case>_<frame>.png images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=5,
        help="Maximum number of columns.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--cell-width", type=float, default=3.0)
    parser.add_argument("--cell-height", type=float, default=3.5)
    args = parser.parse_args()

    rows, columns = grid_shape(len(args.frames), args.columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * args.cell_width, rows * args.cell_height),
        squeeze=False,
    )
    flat_axes = list(axes.flat)

    for axis, frame_index in zip(flat_axes, args.frames):
        path = os.path.join(
            args.input_folder,
            "{}_{:03d}.png".format(args.case, frame_index),
        )
        if not os.path.exists(path):
            raise FileNotFoundError("frame image not found: {}".format(path))

        axis.imshow(plt.imread(path))
        axis.set_title("Frame {}".format(frame_index))
        axis.axis("off")

    for axis in flat_axes[len(args.frames):]:
        axis.remove()

    figure.tight_layout()

    output_folder = os.path.dirname(args.output)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
