import os
import sys
import pickle

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    filename = "creaky_voice/src/main_scripts/outputs/vocal_fold_estimate/results/best_results_08162020.pkl"
    try:
        with open(filename, "rb") as f:
            results = pickle.load(f)
    except OSError as e:
        print(f"OS error: {e}")
        print(f"Failed to load {filename}")

    params = []
    alphas = []
    betas = []
    deltas = []
    for wf in results:
        print(f"Processing {wf}")
        try:
            assert ("alpha" in results[wf]) and (results[wf]["alpha"])
            alpha = results[wf]["alpha"][0]
            beta = results[wf]["beta"][0]
            delta = results[wf]["delta"][0]
            params.append([alpha, beta, delta])
            alphas.append(alpha)
            betas.append(beta)
            deltas.append(delta)
        except AssertionError as e:
            print(e)
            print(f"Skip {wf}")
            continue

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(alphas, deltas, betas, marker="o")
    ax.set_xlabel("alpha")
    ax.set_ylabel("delta")
    ax.set_zlabel("beta")
    plt.show()
