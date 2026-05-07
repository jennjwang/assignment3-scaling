import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    "scaling_laws/results/.mplconfig",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_rows(path):
    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]
    df["status"] = df["status"].astype(str).str.strip()
    return df[
        (df["status"] == "completed")
        & df["last_3_average_loss"].notna()
    ].copy()


def main():
    df = load_rows("scaling_laws/results/data_batch_size_fit.csv")

    plt.figure()
    for batch_size in sorted(df["batch_size"].unique()):
        sub = df[df["batch_size"] == batch_size].sort_values("data_size")
        plt.plot(
            np.log10(sub["data_size"]),
            sub["last_3_average_loss"],
            "o-",
            label=f"batch {batch_size}",
        )

    plt.xlabel("log10 Data Size")
    plt.ylabel("Validation Loss")
    plt.legend()
    os.makedirs("scaling_laws/results", exist_ok=True)
    plt.savefig("scaling_laws/results/data_batch_size_fit.png")
    print("Saved scaling_laws/results/data_batch_size_fit.png")


if __name__ == "__main__":
    main()
