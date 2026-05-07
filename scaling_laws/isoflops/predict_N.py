import pandas as pd
import numpy as np

def power_law(x, a, b):
    return a * x ** b

def calc_compute_budget(runtime_seconds, num_hidden_layers, hidden_size, a, b):
    N = 12 * num_hidden_layers * hidden_size ** 2
    tokens_per_second = a * N**b
    D = tokens_per_second * runtime_seconds
    C = 6 * N * D
    return C

a = 2.25064e+09
b = -0.472

reference_runtime_seconds = 172800
reference_model = (10, 448)

next_C = calc_compute_budget(reference_runtime_seconds, reference_model[0], reference_model[1], a, b)
print(f"next_C: {next_C}")

def main(dataset_path: str):
    df = pd.read_csv(dataset_path)
    df.columns = [column.strip() for column in df.columns]
    df["status"] = df["status"].astype(str).str.strip()
    df = df[
        (df["status"] == "completed")
        & df["last_3_average_loss"].notna()
    ].copy()

    df_grouped = df.groupby(["reference_runtime_seconds", "compute_budget"])["last_3_average_loss"].idxmin()
    df_grouped = df.loc[df_grouped]
    df_grouped = df_grouped.sort_values("compute_budget")
    print("Observed optimal model size by compute budget:")
    for _, row in df_grouped.iterrows():
        print(
            f"{row['reference_runtime_seconds']:g}s ref, "
            f"C={row['compute_budget']:.2e}: "
            f"N*={row['num_params'] / 1_000_000:.2f}M "
            f"({int(row['num_hidden_layers'])} layers, d_model={int(row['hidden_size'])}), "
            f"loss={row['last_3_average_loss']:.4f}, "
            f"status={row['status']}, "
            f"exp={int(row['experiment_id'])}"
        )

    N = df_grouped['num_params']
    D = df_grouped['total_train_tokens']
    C = df_grouped['compute_budget']

    log_C = np.log10(C.to_numpy(dtype=float))
    log_N = np.log10(N.to_numpy(dtype=float))
    log_D = np.log10(D.to_numpy(dtype=float))

    b_N, log_a_N = np.polyfit(log_C, log_N, deg=1)
    a_N = 10 ** log_a_N

    b_D, log_a_D = np.polyfit(log_C, log_D, deg=1)
    a_D = 10 ** log_a_D

    print(f"N*(C) = {a_N:.6g} * C^{b_N:.4f}")
    print(f"D*(C) = {a_D:.6g} * C^{b_D:.4f}")
    print(f"predicted N* at next_C: {power_law(next_C, a_N, b_N):.4g}")
    print(f"predicted D* at next_C: {power_law(next_C, a_D, b_D):.4g}")
    print(f"sanity: 6 * N* * D* = {6 * power_law(next_C, a_N, b_N) * power_law(next_C, a_D, b_D):.4g} vs C = {next_C:.4g}")
    
if __name__ == "__main__":
    main("scaling_laws/isoflops/isoflops.csv")
