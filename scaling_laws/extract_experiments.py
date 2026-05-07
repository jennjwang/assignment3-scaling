import json
from pathlib import Path

import pandas as pd


EXPERIMENTS_PATH = Path("scaling_laws/experiments.json")
OUTPUT_PATH = Path("scaling_laws/results/experiments_batch_lr.csv")


def experiments_to_df(experiments):
    rows = []

    for experiment in experiments:
        config = experiment["training_config"]
        scheduler = config["optimizer_config"]["lr_scheduler"]
        status = experiment["status"]
        val_losses = status.get("val_losses", [])

        rows.append(
            {
                "experiment_id": experiment["experiment_id"],
                "batch_size": config["train_batch_size"],
                "learning_rate": scheduler["peak_value"],
                "total_train_tokens": config["total_train_tokens"],
                "n_evals": config["n_evals"],
                "status": status["status_type"],
                "used_runtime_seconds": status.get("used_runtime_seconds"),
                "final_loss": val_losses[-1] if val_losses else None,
                "last_3_average_loss": (
                    sum(val_losses[-3:]) / min(3, len(val_losses))
                    if val_losses
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


with EXPERIMENTS_PATH.open() as f:
    experiments = json.load(f)

df = experiments_to_df(experiments)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(df)
print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
