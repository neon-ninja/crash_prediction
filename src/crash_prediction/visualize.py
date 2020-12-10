from pathlib import Path

import defopt
import pandas as pd
import hvplot
import hvplot.pandas  # noqa
import panel as pn


def plot_map(dset, varname, title):
    return dset.hvplot.points(
        "X",
        "Y",
        c=varname,
        title=title,
        datashade=True,
        dynspread=True,
        aggregator="mean",
        cmap="fire",
        geo=True,
        tiles="CartoLight",
        frame_width=600,
        frame_height=600,
        groupby="fold",
    )


def display_results(dset_file: Path, *preds_file: Path, show: bool = True):
    """Display accidents according to their severity and compare with predictions

    :param dset_file: CAS dataset .csv file
    :param preds_file: predictions .csv file for one method
    :param show: open the server in a new browser tab on start
    """

    dset = pd.read_csv(dset_file, usecols=["X", "Y", "injuryCrash", "fold"])
    dset["injuryCrash"] = dset["injuryCrash"].astype(float)

    dset["predictions"] = pd.read_csv(preds_file[0])

    crash_map = plot_map(dset, "injuryCrash", "Ground truth")
    preds_map = plot_map(dset, "predictions", "Predictions")

    pane = pn.panel(crash_map + preds_map)
    pn.serve(pane, show=show)


def main():
    defopt.run(display_results)