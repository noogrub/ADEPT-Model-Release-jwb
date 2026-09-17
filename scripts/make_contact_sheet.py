import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

case = "human_disappear_0007_3"
frames = range(140, 145)

fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))

for ax, frame in zip(axes, frames):
    path = "/N/slate/jburgoon/ADEPT-Model-Release/output/default_physics/imgs/{}_{:03d}.png".format(
        case,
        frame,
    )
    image = plt.imread(path)
    ax.imshow(image)
    ax.set_title("Frame {}".format(frame))
    ax.axis("off")

fig.tight_layout()

output_path = (
    "/N/slate/jburgoon/ADEPT-Model-Release/output/svoe_particle_comparison/figures/"
    "human_disappear_0007_3_frames_140_144.png"
)
fig.savefig(output_path, dpi=160)
print(output_path)
