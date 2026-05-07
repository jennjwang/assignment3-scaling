import json
import copy
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from cs336_scaling.client import submit_experiment, get_experiment, get_budget
from cs336_scaling.schemas import SubmitResponse, ExperimentResponse, BudgetSummary
from cs336_scaling.training.training_config import TrainingConfig
import pandas as pd

config_path = "scaling_laws/configs/t1.json"
config = json.load(open(config_path))

"""
For fixed model size N and fixed D:
  sweep B and lr
  record final / last-3-average validation loss
  choose best lr for each B

Then fit:
  lr*(B) = a B^alpha
"""


def wait_for_experiment(experiment_id, poll_seconds=10):
    while True:
        result = get_experiment(experiment_id)
        status_type = result.status.status_type

        if status_type in {"completed", "failed"}:
            return result

        time.sleep(poll_seconds)

def fit_lr_batch_size(config):
    # batch_sizes = [128, 256, 512]
    batch_sizes = [32, 64]
    learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    rows = []

    for bs in batch_sizes:
        for lr in learning_rates:
            config_copy = config.copy()
            config_copy["train_batch_size"] = bs
            config_copy["optimizer_config"]["lr_scheduler"]["peak_value"] = lr

            submit_response = submit_experiment(config_copy)
            print(submit_response)

            result = wait_for_experiment(submit_response.experiment_id)
            print(result)

            if result.status.status_type == "completed":
                val_losses = result.status.val_losses
                rows.append({
                    "batch_size": bs,
                    "learning_rate": lr,
                    "experiment_id": submit_response.experiment_id,
                    "status": "completed",
                    "final_loss": val_losses[-1],
                    "last_3_average_loss": sum(val_losses[-3:]) / min(3, len(val_losses)),
                })
            else:
                rows.append({
                    "batch_size": bs,
                    "learning_rate": lr,
                    "experiment_id": submit_response.experiment_id,
                    "status": "failed",
                    "final_loss": None,
                    "last_3_average_loss": None,
                })

    return pd.DataFrame(rows)


# Path("scaling_laws/results").mkdir(parents=True, exist_ok=True)
# df = fit_lr_batch_size(config)
# df.to_csv("scaling_laws/results/lr_batch_size_fit.csv", index=False)

def fit_data_batch_size(config):
    data_sizes = [8388608, 16777216, 33554432, 67108864]
    batch_sizes = [32, 128, 512]
    anchor_batch_size = 32
    anchor_learning_rate = 3e-3
    seq_len = 512

    rows = []

    for D in data_sizes:
        for bs in batch_sizes:
            learning_rate = anchor_learning_rate * (bs / anchor_batch_size) ** 0.5
            config_copy = copy.deepcopy(config)
            config_copy["train_batch_size"] = bs
            config_copy["total_train_tokens"] = D
            config_copy["optimizer_config"]["lr_scheduler"]["peak_value"] = learning_rate

            submit_response = submit_experiment(config_copy)
            print(submit_response)

            result = wait_for_experiment(submit_response.experiment_id)
            print(result)

            if result.status.status_type == "completed":
                val_losses = result.status.val_losses
                rows.append({
                    "batch_size": bs,
                    "learning_rate": learning_rate,
                    "optimizer_steps": D //(seq_len * bs),
                    "data_size": D,
                    "experiment_id": submit_response.experiment_id,
                    "status": "completed",
                    "final_loss": val_losses[-1],
                    "last_3_average_loss": sum(val_losses[-3:]) / min(3, len(val_losses)),
                })
            else:
                rows.append({
                    "batch_size": bs,
                    "learning_rate": learning_rate,
                    "optimizer_steps": D //(seq_len * bs),
                    "data_size": D,
                    "experiment_id": submit_response.experiment_id,
                    "status": "failed",
                    "final_loss": None,
                    "last_3_average_loss": None,
                })
        
    return pd.DataFrame(rows)

# df = fit_data_batch_size(config)
# df.to_csv("omg scaling_laws/results/data_batch_size_fit_new.csv", index=False)

def fit_model_batch_size(config, results_path="scaling_laws/results/model_batch_size_validate.csv"):
    batch_sizes = [32, 64, 128]
    anchor_learning_rate = 3e-3
    data_size = 33554432
    seq_len = 512
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    model_sizes = [
        (256, 4, 8, 768),      # 6.3M
        (320, 5, 10, 1024),    # 12.3M
        (384, 6, 12, 1024),    # 21.2M
    ]
    rows = []   

    for bs in batch_sizes:
        learning_rate = anchor_learning_rate * (bs / 32) ** 0.5

        for d_model, num_heads, num_hidden_layers, intermediate_size in model_sizes:
            config_copy = copy.deepcopy(config)
            config_copy["architecture_config"]["hidden_size"] = d_model
            config_copy["architecture_config"]["head_dim"] = d_model // num_heads
            config_copy["architecture_config"]["num_attention_heads"] = num_heads
            config_copy["architecture_config"]["num_key_value_heads"] = num_heads
            config_copy["architecture_config"]["num_hidden_layers"] = num_hidden_layers
            config_copy["architecture_config"]["intermediate_size"] = intermediate_size
            config_copy["train_batch_size"] = bs
            config_copy["total_train_tokens"] = data_size
            config_copy["optimizer_config"]["lr_scheduler"]["peak_value"] = learning_rate

            TrainingConfig(**config_copy)

            try:
                submit_response = submit_experiment(config_copy)
                print(submit_response)
            except Exception as e:
                print(e)
                continue

            result = wait_for_experiment(submit_response.experiment_id)
            print(result)

            status_type = result.status.status_type
            if status_type == "completed":
                val_losses = result.status.val_losses
            else:
                reason = getattr(result.status, "reason", None)
                val_losses = getattr(reason, "partial_val_losses", []) or []

            row = {
                "d_model": d_model,
                "num_heads": num_heads,
                "num_hidden_layers": num_hidden_layers,
                "intermediate_size": intermediate_size,
                "model_params": 12 * num_hidden_layers * d_model ** 2,
                "batch_size": bs,
                "learning_rate": learning_rate,
                "optimizer_steps": data_size // (seq_len * bs),
                "data_size": data_size,
                "experiment_id": submit_response.experiment_id,
                "status": status_type,
                "used_runtime_seconds": result.status.used_runtime_seconds,
                "final_loss": val_losses[-1] if val_losses else None,
                "last_3_average_loss": (
                    sum(val_losses[-3:]) / min(3, len(val_losses))
                    if val_losses else None
                ),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_path, index=False)

    return pd.DataFrame(rows)

def fit_model_lr(config, results_path="scaling_laws/results/model_lr_fit.csv"):
    # learning_rates = [1e-3, 3e-3, 1e-2]
    learning_rates = [1e-2, 5e-2, 1e-1]
    # learning_rates = [2e-2, 3e-2]
    batch_size = 32
    data_size = 33554432
    model_sizes = [
        # (256, 4, 8, 768),      # 6.3M
        # (320, 5, 10, 1024),    # 12.3M
        (384, 6, 12, 1024),    # 21.2M
        (448, 7, 14, 1280),    # 29.3M
    ]
    seq_len = 512
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for lr in learning_rates:
        for d_model, num_heads, num_hidden_layers, intermediate_size in model_sizes:
            config_copy = copy.deepcopy(config)
            config_copy["architecture_config"]["hidden_size"] = d_model
            config_copy["architecture_config"]["head_dim"] = d_model // num_heads
            config_copy["architecture_config"]["num_attention_heads"] = num_heads
            config_copy["architecture_config"]["num_key_value_heads"] = num_heads
            config_copy["architecture_config"]["num_hidden_layers"] = num_hidden_layers
            config_copy["architecture_config"]["intermediate_size"] = intermediate_size
            config_copy["train_batch_size"] = batch_size
            config_copy["total_train_tokens"] = data_size
            config_copy["optimizer_config"]["lr_scheduler"]["peak_value"] = lr

            TrainingConfig(**config_copy)

            try:
                submit_response = submit_experiment(config_copy)
                print(submit_response)
            except Exception as e:
                print(e)
                continue

            result = wait_for_experiment(submit_response.experiment_id)
            print(result)

            status_type = result.status.status_type
            if status_type == "completed":
                val_losses = result.status.val_losses
            else:
                reason = getattr(result.status, "reason", None)
                val_losses = getattr(reason, "partial_val_losses", []) or []

            row = {
                "d_model": d_model,
                "num_heads": num_heads,
                "num_hidden_layers": num_hidden_layers,
                "intermediate_size": intermediate_size,
                "model_params": 12 * num_hidden_layers * d_model ** 2,
                "batch_size": batch_size,
                "learning_rate": lr,
                "optimizer_steps": data_size // (seq_len * batch_size),
                "data_size": data_size,
                "experiment_id": submit_response.experiment_id,
                "status": status_type,
                "used_runtime_seconds": result.status.used_runtime_seconds,
                "final_loss": val_losses[-1] if val_losses else None,
                "last_3_average_loss": (
                    sum(val_losses[-3:]) / min(3, len(val_losses))
                    if val_losses else None
                ),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_path, index=False)

    return pd.DataFrame(rows)


def fit_model_lr_focused(config, results_path="scaling_laws/results/model_lr_24m_bracket.csv"):
    learning_rates = [1e-3, 1e-2]
    batch_size = 32
    data_size = 33_554_432
    model_sizes = [
        (448, 7, 10, 1280),    # 24.1M
    ]
    seq_len = 512
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()

    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing.columns = [column.strip() for column in existing.columns]
        for row in existing.itertuples(index=False):
            seen.add((int(row.model_params), int(row.data_size), float(row.learning_rate)))

    for d_model, num_heads, num_hidden_layers, intermediate_size in model_sizes:
        model_params = 12 * num_hidden_layers * d_model ** 2
        for lr in learning_rates:
            key = (model_params, data_size, lr)
            if key in seen:
                print(f"skipping existing N={model_params} D={data_size} lr={lr:g}")
                continue

            config_copy = copy.deepcopy(config)
            config_copy["architecture_config"]["hidden_size"] = d_model
            config_copy["architecture_config"]["head_dim"] = d_model // num_heads
            config_copy["architecture_config"]["num_attention_heads"] = num_heads
            config_copy["architecture_config"]["num_key_value_heads"] = num_heads
            config_copy["architecture_config"]["num_hidden_layers"] = num_hidden_layers
            config_copy["architecture_config"]["intermediate_size"] = intermediate_size
            config_copy["train_batch_size"] = batch_size
            config_copy["total_train_tokens"] = data_size
            config_copy["optimizer_config"]["lr_scheduler"]["peak_value"] = lr

            TrainingConfig(**config_copy)

            try:
                submit_response = submit_experiment(config_copy)
                print(submit_response)
            except Exception as e:
                print(e)
                continue

            result = wait_for_experiment(submit_response.experiment_id)
            print(result)

            status_type = result.status.status_type
            if status_type == "completed":
                val_losses = result.status.val_losses
            else:
                reason = getattr(result.status, "reason", None)
                val_losses = getattr(reason, "partial_val_losses", []) or []

            row = {
                "d_model": d_model,
                "num_heads": num_heads,
                "num_hidden_layers": num_hidden_layers,
                "intermediate_size": intermediate_size,
                "model_params": model_params,
                "batch_size": batch_size,
                "learning_rate": lr,
                "optimizer_steps": data_size // (seq_len * batch_size),
                "data_size": data_size,
                "experiment_id": submit_response.experiment_id,
                "status": status_type,
                "used_runtime_seconds": result.status.used_runtime_seconds,
                "final_loss": val_losses[-1] if val_losses else None,
                "last_3_average_loss": (
                    sum(val_losses[-3:]) / min(3, len(val_losses))
                    if val_losses else None
                ),
            }
            rows.append(row)
            seen.add(key)
            pd.DataFrame([row]).to_csv(
                results_path,
                mode="a",
                header=not results_path.exists(),
                index=False,
            )

    return pd.DataFrame(rows)


def fit_data_lr(config, results_path="scaling_laws/results/data_lr_fit.csv"):
    data_sizes = [
        # 16_777_216,
        # 33_554_432,
        # 67_108_864,
        134_217_728,
    ]
    batch_size = 32
    d_model, num_heads, num_hidden_layers, intermediate_size = (256, 4, 8, 768)
    learning_rates = [
        # 3e-3,
        # 1e-2,
        # 2e-2,
        # 3e-2,
        # 5e-2,
        1e-1,
    ]

    seq_len = 512
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for data_size in data_sizes:
        for lr in learning_rates:
            config_copy = copy.deepcopy(config)
            config_copy["architecture_config"]["hidden_size"] = d_model
            config_copy["architecture_config"]["head_dim"] = d_model // num_heads
            config_copy["architecture_config"]["num_attention_heads"] = num_heads
            config_copy["architecture_config"]["num_key_value_heads"] = num_heads
            config_copy["architecture_config"]["num_hidden_layers"] = num_hidden_layers
            config_copy["architecture_config"]["intermediate_size"] = intermediate_size
            config_copy["train_batch_size"] = batch_size
            config_copy["total_train_tokens"] = data_size
            config_copy["optimizer_config"]["lr_scheduler"]["peak_value"] = lr

            TrainingConfig(**config_copy)

            try:
                submit_response = submit_experiment(config_copy)
                print(submit_response)
            except Exception as e:
                print(e)
                continue

            result = wait_for_experiment(submit_response.experiment_id)
            print(result)

            status_type = result.status.status_type
            if status_type == "completed":
                val_losses = result.status.val_losses
            else:
                reason = getattr(result.status, "reason", None)
                val_losses = getattr(reason, "partial_val_losses", []) or []

            row = {
                "d_model": d_model,
                "num_heads": num_heads,
                "num_hidden_layers": num_hidden_layers,
                "intermediate_size": intermediate_size,
                "model_params": 12 * num_hidden_layers * d_model ** 2,
                "batch_size": batch_size,
                "learning_rate": lr,
                "optimizer_steps": data_size // (seq_len * batch_size),
                "data_size": data_size,
                "experiment_id": submit_response.experiment_id,
                "status": status_type,
                "used_runtime_seconds": result.status.used_runtime_seconds,
                "final_loss": val_losses[-1] if val_losses else None,
                "last_3_average_loss": (
                    sum(val_losses[-3:]) / min(3, len(val_losses))
                    if val_losses else None
                ),
            }
            rows.append(row)
            pd.DataFrame([row]).to_csv(
                results_path,
                mode="a",
                header=not results_path.exists(),
                index=False,
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fit_model_lr_focused(config)
# df.to_csv("scaling_laws/results/throughput_data_fit.csv", index=False)

"""
Best setting seems to be 32 batch size with 3e-3 learning rate.

We see only the slightest throughput improvement with larger batch sizes.
batch  best_lr   best_final_loss   avg_tokens/sec
32     0.003     4.8086            0.772M
64     0.003     5.1777            0.792M
128    0.003     5.7676            0.826M
256    0.003     6.2656            0.787M
512    0.01      6.7480            0.826M

For B=32, we see the throughput worsening with larger model sizes.
384x8    14.2M params   0.927M tokens/s
448x10   24.1M params   0.779M tokens/s
640x12   59.0M params   0.486M tokens/s
768x16   113.2M params  0.304M tokens/s

"""
