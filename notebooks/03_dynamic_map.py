# # Testing dynamic map display
#
# Here we try some dynamic display of all the data points, using [hvplot](https://hvplot.holoviz.org).

from pathlib import Path

import pandas as pd
import hvplot
import hvplot.pandas  # noqa

# First we load the data and binarize the accident severity.

dset_path = Path("..") / "data" / "cas_dataset.csv"
dset = pd.read_csv(dset_path)
dset["badCrash"] = (dset["crashSeverity"] != "Non-Injury Crash") * 1.0

# We exclude Chatham islands to simplify map display. [Datashader](https://datashader.org/)
# is enabled to dynamically aggregate data for display purpose when zooming in
# and out.

dset[dset.X > 0].hvplot.points(
    "X",
    "Y",
    c="badCrash",
    datashade=True,
    dynspread=True,
    cmap="fire",
    geo=True,
    tiles="CartoLight",
    frame_width=600,
    frame_height=600,
)
