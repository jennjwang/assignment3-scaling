import os
import json

os.environ.setdefault(
    "MPLCONFIGDIR",
    "scaling_laws/results/.mplconfig",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_rows(path, data_size=33_554_432, batch_size=32):
    rows = []
    with open(path) as f:
        experiments = json.load(f)

    for experiment in experiments:
        config = experiment["training_config"]
        architecture = config["architecture_config"]
        status = experiment["status"]
        val_losses = status.get("val_losses", [])
        d_model = architecture["hidden_size"]
        num_hidden_layers = architecture["num_hidden_layers"]

        rows.append({
            "d_model": d_model,
            "num_heads": architecture["num_attention_heads"],
            "num_hidden_layers": num_hidden_layers,
            "intermediate_size": architecture["intermediate_size"],
            "model_params": 12 * num_hidden_layers * d_model ** 2,
            "batch_size": config["train_batch_size"],
            "learning_rate": config["optimizer_config"]["lr_scheduler"]["peak_value"],
            "data_size": config["total_train_tokens"],
            "experiment_id": experiment["experiment_id"],
            "status": status["status_type"],
            "used_runtime_seconds": status.get("used_runtime_seconds"),
            "final_loss": val_losses[-1] if val_losses else None,
            "last_3_average_loss": (
                sum(val_losses[-3:]) / min(3, len(val_losses))
                if val_losses else None
            ),
        })

    df = pd.DataFrame(rows)
    return df[
        (df["status"] == "completed")
        & df["last_3_average_loss"].notna()
        & (df["model_params"] > 0)
        & (df["learning_rate"] > 0)
        & (df["data_size"] == data_size)
        & (df["batch_size"] == batch_size)
    ].copy()


def load_csv_rows(path, data_size=33_554_432, batch_size=32):
    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]
    df["status"] = df["status"].astype(str).str.strip()
    return df[
        (df["status"] == "completed")
        & df["last_3_average_loss"].notna()
        & (df["model_params"] > 0)
        & (df["learning_rate"] > 0)
        & (df["data_size"] == data_size)
        & (df["batch_size"] == batch_size)
    ].copy()


def select_best_by_model_size(df):
    best_indices = df.groupby("model_params")["last_3_average_loss"].idxmin()
    best = df.loc[best_indices].sort_values("model_params").copy()

    n_learning_rates = []
    is_bracketed = []
    for row in best.itertuples(index=False):
        lrs = sorted(df[df["model_params"] == row.model_params]["learning_rate"].unique())
        n_learning_rates.append(len(lrs))
        is_bracketed.append(lrs[0] < row.learning_rate < lrs[-1])

    best["n_learning_rates"] = n_learning_rates
    best["is_bracketed"] = is_bracketed
    return best


def fit_lr_model_power_law(best):
    if len(best) < 2:
        return float("nan"), float("nan"), float("nan")

    log_model_params = np.log10(best["model_params"].to_numpy(dtype=float))
    log_learning_rate = np.log10(best["learning_rate"].to_numpy(dtype=float))

    alpha_fit, log_a_fit = np.polyfit(log_model_params, log_learning_rate, deg=1)
    predicted = log_a_fit + alpha_fit * log_model_params
    residuals = log_learning_rate - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_learning_rate - log_learning_rate.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return 10**log_a_fit, alpha_fit, r_squared


def plot_fit(best, a_fit, alpha_fit):
    plt.figure()
    log_model_params = np.log10(best["model_params"])
    log_learning_rate = np.log10(best["learning_rate"])

    plt.plot(log_model_params, log_learning_rate, "o", label="Best")
    if np.isfinite(a_fit) and np.isfinite(alpha_fit):
        log_a_fit = np.log10(a_fit)
        x_range = np.linspace(log_model_params.min(), log_model_params.max(), 100)
        plt.plot(x_range, log_a_fit + alpha_fit * x_range, label="Fit")

    plt.xlabel("log10 Model Size")
    plt.ylabel("log10 Learning Rate")
    plt.legend()
    os.makedirs("scaling_laws/results", exist_ok=True)
    plt.savefig("scaling_laws/results/lr_model_fit.png")


def main():
    dfs = [load_rows("scaling_laws/experiments.json")]
    if os.path.exists("scaling_laws/results/model_lr_24m_bracket.csv"):
        dfs.append(load_csv_rows("scaling_laws/results/model_lr_24m_bracket.csv"))
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates("experiment_id", keep="last")

    os.makedirs("scaling_laws/results", exist_ok=True)
    df.to_csv("scaling_laws/results/model_lr_from_experiments.csv", index=False)

    best = select_best_by_model_size(df)
    best.to_csv("scaling_laws/results/model_lr_best_from_experiments.csv", index=False)
    fit_rows = best[best["is_bracketed"]].copy()
    a_fit, alpha_fit, r_squared = fit_lr_model_power_law(fit_rows)

    print("Best LR by N:")
    for row in best.itertuples(index=False):
        bracket = "bracketed" if row.is_bracketed else "edge"
        print(
            f"  N={int(row.model_params):>10,}  "
            f"lr*={row.learning_rate:g}  "
            f"loss={row.last_3_average_loss:.6f}  "
            f"{row.n_learning_rates} lrs  "
            f"{bracket}"
        )

    if np.isfinite(a_fit) and np.isfinite(alpha_fit):
        print(f"Fit: lr*(N) = {a_fit:.6e} * N^{alpha_fit:.6f}")
        print(f"Log-space R^2: {r_squared:.6f}")
    else:
        print("Need at least two bracketed model-size optima to fit lr*(N).")

    plot_fit(fit_rows if len(fit_rows) else best, a_fit, alpha_fit)
    print("Saved scaling_laws/results/model_lr_from_experiments.csv")
    print("Saved scaling_laws/results/model_lr_best_from_experiments.csv")
    print("Saved scaling_laws/results/lr_model_fit.png")


if __name__ == "__main__":
    main()
