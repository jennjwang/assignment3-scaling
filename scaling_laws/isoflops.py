import json
import copy
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from cs336_scaling.client import submit_experiment, get_experiment, get_budget
from cs336_scaling.training.training_config import TrainingConfig
import pandas as pd

config_path = "scaling_laws/configs/t1.json"
config = json.load(open(config_path))

def wait_for_experiment(experiment_id, poll_seconds=10):
    while True:
        result = get_experiment(experiment_id)
        status_type = result.status.status_type

        if status_type in {"completed", "failed"}:
            return result

        time.sleep(poll_seconds)

def calc_nearest_multiple(x, multiple):
    return int(math.ceil(x / multiple) * multiple)


def calc_compute_budget(runtime_seconds, num_hidden_layers, hidden_size, a, b):
    N = 12 * num_hidden_layers * hidden_size ** 2
    tokens_per_second = a * N**b
    D = tokens_per_second * runtime_seconds
    C = 6 * N * D
    return C

"""
runtime = FLOPs / FLOPs per second
FLOPs = 6 * N * D
runtime = overhead + seconds per token * D

for a 48h run: D = tokens per second * 48h * 60 * 60
D = 

we need to fit the scaling law for the throughput data to estimate C.

10 min -> 600 seconds
30 min -> 1800 seconds
1 hour
3 hours
"""
# a = 5.00976e+09
# b = -0.517

a = 2.47889e+09
b = -0.478

# reference_runtimes = [60, 600, 1200, 1800]
reference_runtimes = [5760]
reference_model = (10, 448)

def build_config(config, D, num_hidden_layers, hidden_size, max_runtime_seconds):

    if max_runtime_seconds > 5000:
        raise ValueError(f"max_runtime_seconds={max_runtime_seconds} must be less than 5000")
    
    config = copy.deepcopy(config)
    batch_size = 32
    learning_rate = 3e-3
    seq_len = 512
    head_dim = 64
    if hidden_size % head_dim != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by head_dim={head_dim}")
    num_heads = hidden_size // head_dim

    tokens_multiple = batch_size * seq_len * config["n_evals"]
    D = calc_nearest_multiple(D, tokens_multiple)

    config["total_train_tokens"] = D
    config["train_batch_size"] = batch_size
    config["optimizer_config"]["lr_scheduler"]["peak_value"] = learning_rate
    config["max_runtime_seconds"] = max_runtime_seconds
    config["architecture_config"]["hidden_size"] = hidden_size
    config["architecture_config"]["num_hidden_layers"] = num_hidden_layers
    config["architecture_config"]["num_attention_heads"] = num_heads
    config["architecture_config"]["num_key_value_heads"] = num_heads
    config["architecture_config"]["head_dim"] = head_dim
    config["architecture_config"]["intermediate_size"] = calc_nearest_multiple(8/3 * hidden_size, 256)

    TrainingConfig(**config)

    return config


def run_model_size(job):
    config = build_config(
        job["base_config"],
        job["total_train_tokens"],
        job["num_hidden_layers"],
        job["hidden_size"],
        job["max_runtime_seconds"],
    )
    
    submit_response = submit_experiment(config)
    print(f"submitted experiment_id={submit_response.experiment_id} N={job['num_params']}")
    result = wait_for_experiment(submit_response.experiment_id)
    print(f"finished experiment_id={submit_response.experiment_id} status={result.status.status_type}")

    status_type = result.status.status_type
    if status_type == "completed":
        val_losses = result.status.val_losses
    else:
        reason = getattr(result.status, "reason", None)
        val_losses = getattr(reason, "partial_val_losses", []) or []

    return {
        "experiment_id": submit_response.experiment_id,
        "status": status_type,
        "reference_runtime_seconds": job["reference_runtime_seconds"],
        "compute_budget": job["compute_budget"],
        "num_params": job["num_params"],
        "num_hidden_layers": job["num_hidden_layers"],
        "hidden_size": job["hidden_size"],
        "num_heads": config["architecture_config"]["num_attention_heads"],
        "intermediate_size": config["architecture_config"]["intermediate_size"],
        "total_train_tokens": config["total_train_tokens"],
        "batch_size": config["train_batch_size"],
        "learning_rate": config["optimizer_config"]["lr_scheduler"]["peak_value"],
        "max_runtime_seconds": job["max_runtime_seconds"],
        "used_runtime_seconds": result.status.used_runtime_seconds,
        "final_loss": val_losses[-1] if val_losses else None,
        "last_3_average_loss": (
            sum(val_losses[-3:]) / min(3, len(val_losses))
            if val_losses else None
        ),
    }

model_sizes = [
    # (4, 128),    # 1.1M
    # (6, 192),    # 2.7M
    # (8, 256),    # 6.3M
    # (10, 320),   # 12.7M
    # (10, 384),   # 17.7M
    # (12, 384),   # 21.2M
    # (12, 448),   # 28.9M
    # (14, 512),   # 44.0M
    # (16, 512),   # 50.3M
    (16, 576),   # 63.7M
    # (18, 640),   # 88.5M
    # (20, 704),   # 118.9M
    # (22, 768),   # 155.0M
]

base_config = config
results_path = Path("scaling_laws/isoflops/isoflops.csv")
results_path.parent.mkdir(parents=True, exist_ok=True)
jobs = []

for runtime_seconds in reference_runtimes:
    reference_num_hidden_layers, reference_hidden_size = reference_model
    C = calc_compute_budget(runtime_seconds, reference_num_hidden_layers, reference_hidden_size, a, b)
    print(f"runtime_seconds: {runtime_seconds}, C: {C}")

    for num_hidden_layers, hidden_size in model_sizes:
        N = 12 * num_hidden_layers * hidden_size ** 2
        D = C / (6 * N)
        tokens_per_second = a * N**b
        tokens_multiple = 32 * 512 * base_config["n_evals"]
        D = calc_nearest_multiple(D, tokens_multiple)
        max_runtime_seconds = math.ceil(1.2 * D / tokens_per_second)
        print(f"N: {N}, D: {D}, C: {C}, max_runtime_seconds: {max_runtime_seconds}")

        jobs.append({
            "base_config": base_config,
            "reference_runtime_seconds": runtime_seconds,
            "compute_budget": C,
            "num_params": N,
            "num_hidden_layers": num_hidden_layers,
            "hidden_size": hidden_size,
            "total_train_tokens": D,
            "max_runtime_seconds": max_runtime_seconds,
        })

if jobs:
    # max_workers = int(os.environ.get("ISOFLOPS_MAX_WORKERS", min(4, len(jobs))))
    max_workers = 4
    print(f"running {len(jobs)} experiments with max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(run_model_size, job): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                row = future.result()
            except Exception as exc:
                print(
                    "failed before writing result "
                    f"N={job['num_params']} layers={job['num_hidden_layers']} "
                    f"hidden_size={job['hidden_size']}: {exc}"
                )
                continue

            pd.DataFrame([row]).to_csv(
                results_path,
                mode="a",
                header=not results_path.exists(),
                index=False,
            )
else:
    print("no experiments to run")
