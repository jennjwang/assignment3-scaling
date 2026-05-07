import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("scaling_laws/results/experiments_batch_lr.csv")
df = df[df["status"] == "completed"]
df = df[df["last_3_average_loss"].notna()]
best = df.loc[df.groupby("batch_size")["last_3_average_loss"].idxmin()]

# log lr* = log a + alpha log B
def log_lr_star(log_B, a, alpha):
    return a + alpha * log_B

log_batch_size = np.log10(best["batch_size"])
log_learning_rate = np.log10(best["learning_rate"])
popt, pcov = curve_fit(log_lr_star, log_batch_size, log_learning_rate)
log_a_fit, alpha_fit = popt
a_fit = 10 ** log_a_fit
print("a_fit, alpha_fit:", a_fit, alpha_fit)

x_range = np.linspace(log_batch_size.min(), log_batch_size.max(), 100)

plt.plot(log_batch_size, log_learning_rate, "o", label="Best")
plt.plot(x_range, log_lr_star(x_range, log_a_fit, alpha_fit), label="Fit")

plt.xlabel("log10 Batch Size")
plt.ylabel("log10 Learning Rate")
plt.legend()
plt.savefig("scaling_laws/results/lr_batch_size_fit.png")