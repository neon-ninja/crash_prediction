# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: crash_prediction
#     language: python
#     name: crash_prediction
# ---

# # Exploration of CAS (Crash analysis system) data

from pathlib import Path

import requests
import pandas as pd
import hvplot.pandas  # NOQA

# ## Data retrieval

# First let's retrieve the dataset from the [Open Data portal](https://opendata-nzta.opendata.arcgis.com/datasets/crash-analysis-system-cas-data-1).
# Multiple file formats are available (csv, kml, geojson, ...), the most compact
# being the .csv one.

dset_path = Path("..") / "data" / "Crash_Analysis_System__CAS__Data.csv"

if not dset_path.exists():
    dset_path.parent.mkdir(exist_ok=True, parents=True)
    dset_url = "https://opendata.arcgis.com/datasets/8d684f1841fa4dbea6afaefc8a1ba0fc_0.csv?outSR=%7B%22latestWkid%22%3A2193%2C%22wkid%22%3A2193%7D"
    dset_web = requests.get(dset_url)
    with dset_path.open("wb") as fd:
        fd.write(dset_web.content)

# Next we load the data and have a quick look to check if there no obvious
# loading error.

dset = pd.read_csv(dset_path)
dset

# The dataset contains 72 columns, describing various aspects of the recorded
# car crashes. The full description of the fields is available online, see
# https://opendata-nzta.opendata.arcgis.com/pages/cas-data-field-descriptions.

dset.columns

# Note that `X` and `Y` are geographical coordinates using NZTM2000 (New Zealand
# Transverse Mercator 2000) coordinate system (see [EPSG:2193](https://epsg.io/2193)).

# ## Spatio-temporal aspects

hv_map = dset.hvplot.points(
    "X",
    "Y",
    tiles="OSM",
    crs=2193,
    frame_width=400,
    datashade=True,
)
hv_map

hv_auckland = dset.hvplot.points(
    "X",
    "Y",
    tiles="OSM",
    crs=2193,
    frame_width=400,
    datashade=True,
)
hv_auckland.redim.range(X=(1739669.89, 1781221.41), Y=(5932097.90, 5901931.60))

hv_cbd = dset.hvplot.points(
    "X",
    "Y",
    tiles="OSM",
    crs=2193,
    frame_width=400,
    datashade=True,
)
hv_cbd.redim.range(X=(1755876.21,  1758568.09), Y=(5921526.71, 5918933.89))

# +
# import geopandas as gpd
#
# gdset = gpd.GeoDataFrame(
#     dset, geometry=gpd.points_from_xy(dset["X"], dset["Y"], crs=2193)
# )
