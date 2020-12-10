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
        frame_width=450,
        frame_height=450,
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

    if preds_file:
        filename = pn.widgets.Select(name="Filename", options=list(preds_file))

        @pn.depends(filename=filename.param.value)
        def plot_crash_n_results(filename):
            dset["predictions"] = pd.read_csv(preds_file[0])
            dset["error"] = dset["injuryCrash"] - dset["predictions"]

            crash_map = plot_map(dset, "injuryCrash", "Ground truth")
            preds_map = plot_map(dset, "predictions", "Predictions")
            error_map = plot_map(dset, "error", "Errors")
            maps = crash_map + preds_map + error_map

            print(maps)
            return maps

        pane = pn.panel(pn.Column(filename, plot_crash_n_results))

    else:
        pane = pn.panel(plot_map(dset, "injuryCrash", "Ground truth"))

    pn.serve(pane, show=show)


def main():
    defopt.run(display_results)
