from pathlib import Path

import defopt
import pandas as pd
import hvplot
import hvplot.pandas  # noqa
import panel as pn


def display_results(dset_file: Path, *preds_file: Path, show: bool = True):
    """Display accidents according to their severity and compare with predictions

    :param dset_file: CAS dataset .csv file
    :param preds_file: predictions .csv file for one method
    :param show: open the server in a new browser tab on start
    """

    dset = pd.read_csv(dset_file, usecols=["X", "Y", "injuryCrash", "fold"])
    dset["injuryCrash"] = dset["injuryCrash"].astype(float)

    crash_map = dset.hvplot.points(
        "X",
        "Y",
        c="injuryCrash",
        datashade=True,
        dynspread=True,
        cmap="fire",
        geo=True,
        tiles="CartoLight",
        frame_width=600,
        frame_height=600,
    )

    pane = pn.panel(crash_map)
    pn.serve(pane, show=show)


def main():
    defopt.run(display_results)