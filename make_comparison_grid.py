"""Build a 3-column comparison grid in the style of the original repo:
legitimate examples | perturbed (adversarial) examples | changed-prediction map.

Rows are the epsilon values swept in the untargeted attack.
"""
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image

EPSILONS = [0.03, 0.05, 0.1, 0.2, 0.3]
OUTPUT_DIR = "output/cputorch"
COLUMNS = [
    ("legitimate", "Clean"),
    ("perturbed", "Adversarial"),
    ("changed", "Changed predictions"),
]


def image_path(prefix, eps):
    # solver.py builds the name with plain str(epsilon): e_0.1, not e_0.10
    return os.path.join(OUTPUT_DIR, "{}(t_-1,e_{},i_1).jpg".format(prefix, eps))


fig, axes = plt.subplots(len(EPSILONS), len(COLUMNS), figsize=(9, 15))
missing = []

for row, eps in enumerate(EPSILONS):
    for col, (prefix, title) in enumerate(COLUMNS):
        ax = axes[row, col]
        path = image_path(prefix, eps)
        if os.path.exists(path):
            ax.imshow(Image.open(path))
        else:
            missing.append(path)
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=11)
    axes[row, 0].set_ylabel("ε = {}".format(eps), fontsize=11)

plt.tight_layout()
plt.savefig("comparison_grid.png", dpi=150, bbox_inches="tight")

if missing:
    print("missing images:")
    for path in missing:
        print("  " + path)
    sys.exit(1)

print("saved: comparison_grid.png")
