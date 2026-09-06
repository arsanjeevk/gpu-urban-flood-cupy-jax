"""Build the report's secondary-benchmark figure from verified CSV evidence."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "../../HF_CUPY_JAX_COMPLETED_PROJECT/benchmarks/secondary_60min/benchmark_results.csv"
).resolve()
OUTPUT = HERE / "figures/19_runtime_distribution.png"


def main() -> None:
    measured: dict[str, list[float]] = {"cupy": [], "jax": []}
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] == "measured":
                measured[row["backend"]].append(float(row["solver_runtime_s"]))

    if {key: len(value) for key, value in measured.items()} != {"cupy": 10, "jax": 10}:
        raise RuntimeError("Expected exactly ten measured runtimes per backend")

    cupy = np.asarray(measured["cupy"])
    jax = np.asarray(measured["jax"])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)

    positions = np.arange(1, 11)
    axes[0].plot(positions, cupy, "o-", color="#1769aa", label="CuPy")
    axes[0].plot(positions, jax, "s-", color="#c43c39", label="JAX")
    axes[0].set_xlabel("Measured repetition")
    axes[0].set_ylabel("Solver runtime (s)")
    axes[0].set_xticks(positions)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    box = axes[1].boxplot(
        [cupy, jax],
        tick_labels=["CuPy", "JAX"],
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, colour in zip(box["boxes"], ["#74add1", "#e58b87"], strict=True):
        patch.set_facecolor(colour)
    axes[1].scatter(
        np.repeat([1, 2], 10) + np.tile(np.linspace(-0.07, 0.07, 10), 2),
        np.concatenate([cupy, jax]),
        color="#303030",
        s=15,
        zorder=3,
    )
    axes[1].set_ylabel("Solver runtime (s)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_title("3 warm-ups + 10 measured runs per backend")

    fig.suptitle("Secondary 60-minute production benchmark", fontweight="bold")
    fig.savefig(OUTPUT, dpi=220, facecolor="white")


if __name__ == "__main__":
    main()
