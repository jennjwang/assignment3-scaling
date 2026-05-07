import os
from pathlib import Path
import pandas as pd


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".mplconfig"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(dataset_path: str):
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path)
    df = df[df["last_3_average_loss"].notna()]

    plt.figure(figsize=(8, 5))
    for (runtime_seconds, compute_budget), group in df.groupby(["reference_runtime_seconds", "compute_budget"]):
        group = group.sort_values("num_params")
        print(group)

        x = group["num_params"].to_numpy(dtype=float) / 1_000_000
        y = group['last_3_average_loss'].to_numpy(dtype=float)

        line = plt.plot(x, y, "-", label=f"{runtime_seconds:g}s ref, C={compute_budget:.2e}")[0]
        color = line.get_color()

        completed = group[group["status"] == "completed"]
        failed = group[group["status"] != "completed"]
        if len(completed) > 0:
            plt.scatter(
                completed["num_params"].to_numpy(dtype=float) / 1_000_000,
                completed["last_3_average_loss"].to_numpy(dtype=float),
                color=color,
                zorder=3,
            )
        if len(failed) > 0:
            plt.scatter(
                failed["num_params"].to_numpy(dtype=float) / 1_000_000,
                failed["last_3_average_loss"].to_numpy(dtype=float),
                facecolors="none",
                edgecolors=color,
                zorder=3,
            )
    
        log_x = np.log10(x)
        a, b, c = np.polyfit(log_x, y, deg=2)

        log_x_span = log_x.max() - log_x.min()
        log_x_fit = np.linspace(
            log_x.min() - 0.05 * log_x_span,
            log_x.max() + 0.25 * log_x_span,
            200,
        )
        x_fit = 10 ** log_x_fit
        y_fit = np.polyval([a, b, c], log_x_fit)

        plt.plot(x_fit, y_fit, "--", color=color)

        if a > 0:
            log_x_min = -b / (2 * a)
            if log_x.min() <= log_x_min <= log_x.max():
                x_min = 10 ** log_x_min
                y_min = np.polyval([a, b, c], log_x_min)

                plt.plot(x_min, y_min, "x", color=color, markersize=10)
                print(f"C={compute_budget:.2e}: min at {x_min:.2f}M params, loss={y_min:.4f}")


    plt.xscale("log")
    plt.xlabel("Number of Parameters (M)")
    plt.ylabel("Last 3 Average Loss")
    plt.title("IsoFLOPs Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dataset_path.with_name("isoflop_curve.png"), dpi=200)


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "isoflops.csv")
