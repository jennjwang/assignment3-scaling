from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * x**b

def linear(x, m, c):
    return m * x + c

def runtime_from_throughput(model_params, data_size, a, b, overhead=0.0):
    tokens_per_second = power_law(model_params, a, b)
    return overhead + data_size / tokens_per_second

def fit_throughput(df):
    df = df.copy()
    df["seconds_per_token"] = df["runtime_seconds"] / df["data_size"]

    fit_points = (
        df.groupby("model_params", as_index=False)
        .agg(
            tokens_per_second=("tokens_per_second", "median"),
            seconds_per_token=("seconds_per_token", "median"),
            num_runs=("tokens_per_second", "size"),
        )
        .sort_values("model_params")
    )

    model_params = fit_points["model_params"].to_numpy(dtype=float)
    tokens_per_second = fit_points["tokens_per_second"].to_numpy(dtype=float)
    seconds_per_token = fit_points["seconds_per_token"].to_numpy(dtype=float)

    power_popt, _ = curve_fit(power_law, model_params, tokens_per_second)
    a_fit, b_fit = power_popt

    linear_popt, _ = curve_fit(linear, model_params, seconds_per_token)
    m_fit, c_fit = linear_popt

    return fit_points, a_fit, b_fit, m_fit, c_fit


throughput_df = pd.read_csv("scaling_laws/results/throughput.csv")
throughput_df = throughput_df[
    (throughput_df["status"] == "completed") & (throughput_df["batch_size"] == 32)
].copy()
throughput_df = throughput_df[
    ["experiment_id", "model_params", "data_size", "runtime_seconds", "tokens_per_second"]
]
throughput_df["source"] = "throughput"

isoflops_df = pd.read_csv("scaling_laws/isoflops/isoflops.csv")
isoflops_df = isoflops_df[
    (isoflops_df["status"] == "completed")
    & (isoflops_df["batch_size"] == 32)
    & isoflops_df["used_runtime_seconds"].notna()
].copy()
isoflops_df = isoflops_df.rename(
    columns={
        "num_params": "model_params",
        "total_train_tokens": "data_size",
        "used_runtime_seconds": "runtime_seconds",
    }
)
isoflops_df["tokens_per_second"] = isoflops_df["data_size"] / isoflops_df["runtime_seconds"]
isoflops_df = isoflops_df[
    ["experiment_id", "model_params", "data_size", "runtime_seconds", "tokens_per_second"]
]
isoflops_df["source"] = "isoflops"

df = pd.concat([throughput_df, isoflops_df], ignore_index=True)
df = df.dropna(subset=["model_params", "data_size", "runtime_seconds", "tokens_per_second"])

fit_df, a_fit, b_fit, m_fit, c_fit = fit_throughput(df)
Path("scaling_laws/results").mkdir(parents=True, exist_ok=True)
df.sort_values(["model_params", "data_size"]).to_csv(
    "scaling_laws/results/throughput_fit_data.csv",
    index=False,
)
fit_df.to_csv("scaling_laws/results/throughput_fit_points.csv", index=False)

print(f"using {len(df)} completed B=32 runs across {len(fit_df)} model sizes")
print(fit_df)
print(f"power:  tokens/sec = {a_fit:.6g} * model_params^{b_fit:.3f}")
print(f"linear: seconds/token = {m_fit:.6g} * model_params + {c_fit:.6g}")
