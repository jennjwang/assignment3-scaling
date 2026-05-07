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
        & (df["data_size"] > 0)
        & (df["learning_rate"] > 0)
    ].copy()


def select_best_by_data_size(df):
    best_indices = df.groupby("data_size")["last_3_average_loss"].idxmin()
    best = df.loc[best_indices].sort_values("data_size").copy()

    bracketed = []
    for row in best.itertuples(index=False):
        group = df[df["data_size"] == row.data_size].sort_values("learning_rate")
        lrs = group["learning_rate"].to_numpy()
        best_position = np.where(lrs == row.learning_rate)[0][0]
        bracketed.append(0 < best_position < len(lrs) - 1)

    best["is_bracketed"] = bracketed
    return best


def fit_lr_data_power_law(best):
    log_data_size = np.log10(best["data_size"].to_numpy(dtype=float))
    log_learning_rate = np.log10(best["learning_rate"].to_numpy(dtype=float))

    alpha_fit, log_a_fit = np.polyfit(log_data_size, log_learning_rate, deg=1)
    predicted = log_a_fit + alpha_fit * log_data_size
    residuals = log_learning_rate - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_learning_rate - log_learning_rate.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return 10**log_a_fit, alpha_fit, r_squared


def plot_fit(best, a_fit, alpha_fit):
    log_data_size = np.log10(best["data_size"])
    log_learning_rate = np.log10(best["learning_rate"])
    log_a_fit = np.log10(a_fit)
    x_range = np.linspace(log_data_size.min(), log_data_size.max(), 100)

    plt.plot(log_data_size, log_learning_rate, "o", label="Best")
    plt.plot(x_range, log_a_fit + alpha_fit * x_range, label="Fit")

    plt.xlabel("log10 Data Size")
    plt.ylabel("log10 Learning Rate")
    plt.legend()
    os.makedirs("scaling_laws/results", exist_ok=True)
    plt.savefig("scaling_laws/results/lr_data_fit.png")


def main():
    df = load_rows("scaling_laws/results/data_lr_fit.csv")
    best = select_best_by_data_size(df)
    a_fit, alpha_fit, r_squared = fit_lr_data_power_law(best)

    print("Best LR by D:")
    for row in best.itertuples(index=False):
        bracket = "bracketed" if row.is_bracketed else "edge"
        print(
            f"  D={int(row.data_size):>10,}  "
            f"lr*={row.learning_rate:g}  "
            f"loss={row.last_3_average_loss:.6f}  "
            f"{bracket}"
        )

    print(f"Fit: lr*(D) = {a_fit:.6e} * D^{alpha_fit:.6f}")
    print(f"Log-space R^2: {r_squared:.6f}")

    plot_fit(best, a_fit, alpha_fit)
    print("Saved scaling_laws/results/lr_data_fit.png")


if __name__ == "__main__":
    main()
