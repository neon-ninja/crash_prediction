from pathlib import Path

import defopt
import pandas as pd
import seaborn as sb
import sklearn.metrics as metrics
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def plot_confusion_matrix(y, y_pred, ax=None):
    labels = ["Non-Injury Crashes", "Injury Crashes"]
    conf_mtx = metrics.confusion_matrix(y, y_pred)
    return sb.heatmap(
        conf_mtx / conf_mtx.sum(),
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        fmt=".2%",
        annot=True,
        ax=ax,
    )


def plot_calibration_curves(y, y_prob):
    fig, ax = plt.subplots(figsize=(10, 7))

    prob_true, prob_pred = calibration_curve(y, y_prob)
    ax.plot([0, 1], [0, 1], "--k")
    ax.plot(prob_true, prob_pred, "-o")
    ax.set_title("Calibration curve")
    ax.set_xlabel("true probabilities")
    ax.set_ylabel("predicted probabilities")

    fig.tight_layout()

    return fig


def evaluate(dset_file: Path, preds_file: Path, output_folder: Path):
    """Score and plots results

    :param dset_file: CAS dataset .csv file
    :param pred_file: predictions .csv file
    :param output_folder: output folder for the figures
    """
    dset = pd.read_csv(dset_file)

    y_prob = pd.read_csv(preds_file)
    y_pred = y_prob > 0.5

    output_folder.mkdir(parents=True, exist_ok=True)

    for fold, group in dset.groupby("fold"):
        y = group["injuryCrash"]

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_confusion_matrix(y, y_pred.loc[y.index], ax)
        fig.savefig(output_folder / f"confusion_matrix__{fold}.png")

        fig = plot_calibration_curves(y, y_prob.loc[y.index])
        fig.savefig(output_folder / f"calibration_curves__{fold}.png")


def main():
    defopt.run(evaluate)
