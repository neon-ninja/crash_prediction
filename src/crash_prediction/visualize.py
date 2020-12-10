from pathlib import Path

import defopt
import pandas as pd
import hvplot
import hvplot.pandas  # noqa
import panel as pn


def plot_map(dset, varname, title, cmap="fire", **kwargs):
    return dset.hvplot.points(
        "X",
        "Y",
        c=varname,
        title=title,
        rasterize=True,
        aggregator="mean",
        cmap=cmap,
        geo=True,
        tiles="CartoLight",
        frame_width=450,
        frame_height=450,
        groupby="fold",
        **kwargs
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
        filename_widget = pn.widgets.Select(
            name="Predictions file",
            options=list(preds_file),
            margin=(20, 20, 0, 20),
            width=400,
        )

        @pn.depends(filename=filename_widget.param.value)
        def plot_crash_n_results(filename):
            dset["predictions"] = pd.read_csv(filename)
            dset["error"] = dset["injuryCrash"] - dset["predictions"]

            crash_map = plot_map(dset, "injuryCrash", "Ground truth", clim=(0, 1))
            preds_map = plot_map(dset, "predictions", "Predictions", clim=(0, 1))
            error_map = plot_map(dset, "error", "Errors", cmap="seismic", clim=(-1, 1))

            hv_maps = pn.panel(crash_map + preds_map + error_map)
            return pn.Column(hv_maps[1][0][0], hv_maps[0])

        pane = pn.Column(filename_widget, plot_crash_n_results)

    else:
        pane = pn.panel(plot_map(dset, "injuryCrash", "Ground truth"))

    pn.serve(pane, show=show)


def main():
    defopt.run(display_results)
