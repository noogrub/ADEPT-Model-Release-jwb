"""Inspect ADEPT particle-level scores and locations at selected frames."""

import argparse
import json
import os

from svoe_calibration.particle_analysis import inspect_result_frames


def main():
    parser = argparse.ArgumentParser(
        description="Inspect particle-level data in one ADEPT result."
    )
    parser.add_argument(
        "--result",
        required=True,
        help="Path to a particle-experiment result.json file.",
    )
    parser.add_argument(
        "--case",
        help="Case name; inferred when the result contains one case.",
    )
    parser.add_argument(
        "--frames",
        required=True,
        nargs="+",
        type=int,
        help="Zero-based frame indices to inspect.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. The report is always printed.",
    )
    args = parser.parse_args()

    report = inspect_result_frames(
        args.result,
        args.frames,
        case_name=args.case,
    )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)

    if args.output:
        parent = os.path.dirname(args.output)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "w") as stream:
            stream.write(rendered)
            stream.write("\n")


if __name__ == "__main__":
    main()
