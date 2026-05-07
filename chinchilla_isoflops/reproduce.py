"""
Write a script to reproduce the IsoFLOPs method described above for fitting scaling laws using
the final training loss from a set of training runs. For this problem, use the (synthetic) data from
training runs given in the file data/isoflops_curves.json. This file contains a JSON array, where
each element is an object describing a training run. Here are the first two runs for illustrating the
format:
"""

import json
import os
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main(dataset_path: str):
    """
    N = parameters (model size)
    D = tokens (dataset size)
    C = compute budget
    """

    with open(dataset_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # for item in data:
    #     print(item)
    
    def power_law(x, a, b):
        return a * x ** b

    def log_power_law(log_x, log_a, b):
        return log_a + b * log_x

    """
    Deliverable: A plot showing your scaling law for model size by compute budget, showing the
    data points used to fit the scaling law and extrapolating up to at least 1024 FLOPs. Then, a
    one-sentence response with your predicted optimal model size.
    """
    df_grouped = df.groupby('compute_budget').idxmin()['final_loss']
    df_grouped = df.loc[df_grouped]
    print(df_grouped)

    N = df_grouped['parameters']
    C = df_grouped['compute_budget']

    popt, pcov = scipy.optimize.curve_fit(log_power_law, np.log(C), np.log(N))
    log_a_fit, b_fit = popt
    a_fit = np.exp(log_a_fit)
    print("a_fit, b_fit:", a_fit, b_fit)
    print("predicted optimal model size at 10^23 FLOPs:", power_law(1e23, a_fit, b_fit))
    print("predicted optimal model size at 10^24 FLOPs:", power_law(1e24, a_fit, b_fit))

    """
    a_fit, b_fit: 1.1626... 0.4687...
    predicted optimal model size at 10^23 FLOPs: 7.005e10
    predicted optimal model size at 10^24 FLOPs: 2.061e11
    """

    x_range = np.logspace(np.log10(C.min()), 24, 100)
    plt.plot(C, N, 'o', label='Data')
    plt.plot(x_range, power_law(x_range, a_fit, b_fit), label='Fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Compute Budget (FLOPs)')
    plt.ylabel('Model Size (Parameters)')
    plt.title('Scaling Law for Model Size by Compute Budget')
    plt.savefig('model_size_by_compute_budget.png')

    """
    Deliverable: A plot showing your scaling law for dataset size by compute budget, showing
    the data points used to fit the scaling law and extrapolating up to at least 1024 FLOPs.
    Then, a one-sentence response with your predicted optimal dataset size.
    """

    D = C / (6 * N)

    popt, pcov = scipy.optimize.curve_fit(log_power_law, np.log(C), np.log(D))
    log_a_fit, b_fit = popt
    a_fit = np.exp(log_a_fit)
    print("a_fit, b_fit:", a_fit, b_fit)
    print("predicted optimal dataset size at 10^23 FLOPs:", power_law(1e23, a_fit, b_fit))
    print("predicted optimal dataset size at 10^24 FLOPs:", power_law(1e24, a_fit, b_fit))

    plt.clf()
    plt.plot(C, D, 'o', label='Data')
    plt.plot(x_range, power_law(x_range, a_fit, b_fit), label='Fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Compute Budget (FLOPs)')
    plt.ylabel('Dataset Size (Tokens)')
    plt.title('Scaling Law for Dataset Size by Compute Budget')
    plt.savefig('dataset_size_by_compute_budget.png')

    """
    a_fit, b_fit: 0.1433... 0.5313...
    predicted optimal dataset size at 10^23 FLOPs: 2.379e11
    predicted optimal dataset size at 10^24 FLOPs: 8.086e11
    """


if __name__ == "__main__":
    main("data/isoflops_curves.json")
