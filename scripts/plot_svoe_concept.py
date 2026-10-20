"""Create a slide-ready ADEPT-to-SVoE concept figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from svoe_calibration.calibration import calibrate_scores


RESULT_ROOT = Path("output/default_physics/results")
OUTPUT = Path(
    "output/svoe_calibration/figures/"
    "svoe_concept_human_block.png"
)

CASES = [
    "human_block_0051_0",
    "human_disappear_0007_3",
]

TARGET_CASE = "human_block_0051_0"


def load_scores(case_name):
    path = RESULT_ROOT / "{}.json".format(case_name)

    with path.open("r") as stream:
        payload = json.load(stream)

    # ADEPT result files store the case under its case name.
    if case_name in payload:
        payload = payload[case_name]

    return [float(value) for value in payload["all"]]


# Reproduce our provisional reference:
# two cases × 170 frames = 340 reference scores.
scores_by_case = {
    case_name: load_scores(case_name)
    for case_name in CASES
}

reference_scores = []
for case_name in CASES:
    reference_scores.extend(scores_by_case[case_name])

adept_scores = scores_by_case[TARGET_CASE]

calibrated = calibrate_scores(
    adept_scores,
    reference_scores,
)

svoe_bits = [
    record["svoe_bits"]
    for record in calibrated
]

peak_frame = max(
    range(len(adept_scores)),
    key=lambda frame: adept_scores[frame],
)

peak_adept = adept_scores[peak_frame]
peak_record = calibrated[peak_frame]
peak_svoe = peak_record["svoe_bits"]
peak_probability = peak_record["tail_probability"]
rarity = 1.0 / peak_probability

frames = list(range(len(adept_scores)))

fig, axes = plt.subplots(
    2,
    1,
    figsize=(13.333, 7.5),
    sharex=True,
)

fig.suptitle(
    "SVoE Gives ADEPT Surprise an Interpretable Scale",
    fontsize=20,
    y=0.97,
)

# ADEPT panel
axes[0].plot(
    frames,
    adept_scores,
    color="#2774B8",
    linewidth=1.8,
)

axes[0].axvline(
    peak_frame,
    color="#555555",
    linestyle="--",
    linewidth=1,
    alpha=0.7,
)

axes[0].scatter(
    [peak_frame],
    [peak_adept],
    color="#2774B8",
    s=45,
    zorder=3,
)

axes[0].annotate(
    "ADEPT peak = {:.2f}".format(peak_adept),
    xy=(peak_frame, peak_adept),
    xytext=(-125, -18),
    textcoords="offset points",
    fontsize=12,
    arrowprops={
        "arrowstyle": "->",
        "color": "#333333",
    },
)

axes[0].set_ylabel(
    "ADEPT score",
    fontsize=13,
)

axes[0].set_title(
    "Original model-specific score",
    fontsize=14,
)

# SVoE panel
axes[1].plot(
    frames,
    svoe_bits,
    color="#D64541",
    linewidth=1.8,
)

axes[1].axvline(
    peak_frame,
    color="#555555",
    linestyle="--",
    linewidth=1,
    alpha=0.7,
)

axes[1].scatter(
    [peak_frame],
    [peak_svoe],
    color="#D64541",
    s=45,
    zorder=3,
)

axes[1].annotate(
    (
        "SVoE peak = {:.2f} bits\n"
        "p = {:.4f}  (about 1 in {:.0f} frames)"
    ).format(
        peak_svoe,
        peak_probability,
        rarity,
    ),
    xy=(peak_frame, peak_svoe),
    xytext=(-190, -35),
    textcoords="offset points",
    fontsize=12,
    arrowprops={
        "arrowstyle": "->",
        "color": "#333333",
    },
)

axes[1].set_ylabel(
    "SVoE surprise (bits)",
    fontsize=13,
)

axes[1].set_xlabel(
    "Video frame",
    fontsize=13,
)

axes[1].set_title(
    "The same result expressed as empirical rarity",
    fontsize=14,
)

for axis in axes:
    axis.grid(
        axis="y",
        color="#DDDDDD",
        linewidth=0.8,
        alpha=0.8,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=11)

fig.text(
    0.5,
    0.025,
    (
        "Prototype calibration using two reproduced "
        "170-frame ADEPT sequences. "
        "Final calibration will use designated control videos."
    ),
    ha="center",
    fontsize=10,
    color="#555555",
)

fig.tight_layout(
    rect=(0.03, 0.065, 0.98, 0.94),
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    str(OUTPUT),
    dpi=200,
    facecolor="white",
    bbox_inches="tight",
)

print(OUTPUT)
print("peak frame:", peak_frame)
print("ADEPT peak:", peak_adept)
print("SVoE peak:", peak_svoe)
print("tail probability:", peak_probability)
print("rarity: 1 in {:.1f}".format(rarity))
