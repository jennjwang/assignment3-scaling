import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("scaling_laws/results/experiments_batch_lr.csv")
df = df[(df["status"] == "completed") & df["last_3_average_loss"].notna()]

# Pick one data size so the heatmap is comparable.
D = 16_777_216
df = df[df["total_train_tokens"] == D]

grid = df.pivot_table(
    index="batch_size",
    columns="learning_rate",
    values="last_3_average_loss",
    aggfunc="min",
)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(grid.values, aspect="auto", cmap="YlGnBu")

ax.set_xticks(range(len(grid.columns)))
ax.set_xticklabels([f"{lr:g}" for lr in grid.columns], rotation=45, ha="right")

ax.set_yticks(range(len(grid.index)))
ax.set_yticklabels([str(bs) for bs in grid.index])

ax.set_xlabel("Learning Rate")
ax.set_ylabel("Batch Size")
ax.set_title(f"Validation Loss Grid, D={D:,} tokens")

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        value = grid.iloc[i, j]
        if pd.notna(value):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)

fig.colorbar(im, ax=ax, label="Last-3 Avg Val Loss")
plt.tight_layout()
plt.savefig("scaling_laws/results/lr_batch_size_heatmap.png", dpi=200)
