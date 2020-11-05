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

import itertools as it
from pathlib import Path

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import hvplot.pandas  # NOQA

# set seaborn default style
sb.set()

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

# ## Spatial features
#
# First, we will look at the location of the crashes. More accidents happen in
# densier areas and it would be good to compare with population density.

# common parameters for the maps
map_kwargs = {
    "tiles": "CartoLight",
    "crs": 2193,
    "datashade": True,
    "dynspread": True,
    "cmap": "fire",
}

dset.hvplot.points("X", "Y", frame_width=500, **map_kwargs)

# In dense aread, like in Auckland CBD, there are enough crashes events to map
# the local road network.

bbox_cbd = {"X": (1755876.21, 1758568.09), "Y": (5921526.71, 5918933.89)}
dset.hvplot.points("X", "Y", frame_width=500, **map_kwargs).redim.range(**bbox_cbd)

# At a coarser level, there is also the region information.

region_perc = dset["region"].value_counts(normalize=True)
ax = region_perc.plot.bar(ylabel="fraction of crashes", figsize=(10, 5))
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

print(
    f"The top 4 regions account for {region_perc.nlargest(4).sum() * 100:0.1f}% of the crashes."
)

# ## Temporal features
#
# The dataset contains few temporal features:
#
# - `crashYear` and `crashFinancialYear`, respectively the year and final year
#   of each crash,
# - `holiday`, whether it occurs during a holiday period.
#
# So we won't be able to study daily, weekly and yearly patterns with these data.
#
# If we look at the yearly counts, we can see some fluctuations, mostly driven
# by Auckland region but still noticeable in other parts of the country.
# Year 2020 is much lower as it's the current year.

year_counts = dset["crashYear"].value_counts(sort=False)
year_counts.plot.bar(ylabel="# crashes", figsize=(10, 5))

year_region_counts = (
    dset.groupby(["crashYear", "region"]).size().reset_index(name="# crashes")
)
_, ax = plt.subplots(figsize=(10, 5))
sb.pointplot(data=year_region_counts, x="crashYear", y="# crashes", hue="region", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
_ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

# We can also explore the spatio-temporal patterns too. Here we focus on
# Auckland's CBD.

hv_cbd_year = dset.hvplot.points(
    "X",
    "Y",
    groupby="crashYear",
    frame_width=500,
    **map_kwargs,
)
hv_cbd_year.redim.range(**bbox_cbd)

# The other temporal attribute is the holiday. Chistmas is the holiday period
# with most of the accidents. How the period is computed is not clear, so the
# larger amount of accident could be partly due to the time extent.

holiday_counts = dset["holiday"].fillna("Normal day").value_counts()
ax = holiday_counts.plot.bar(ylabel="# crashes", figsize=(10, 5), rot=0)
_ = ax.set(yscale="log")

# ## Road Features
#
# From the dataset fields description, the following features seem specific to
# the type of road:
#
# - `flatHill`, whether the road is flat or sloped,
# - `junctionType`, type of junction the crash happened at (may also be *unknown*
#   & crashes not occurring at a junction are also *unknown*),
# - `NumberOfLanes`, number of lanes on the crash road,
# - `roadCharacter`, general nature of the road,
# - `roadCurvature`, simplified curvature of the road,
# - `roadLane`, lane configuration of the road (' ' for unknown or invalid
#   configurations),
# - `roadMarkings`, road markings at the crash site,
# - `roadSurface`, road surface description applying at the crash site,
# - `speedLimit`,  speed limit in force at the crash site at the time of the
#   crash (number, or 'LSZ' for a limited speed zone),
# - `streetLight`, street lighting at the time of the crash, **note this is also
#   a sort of temporal information**,
# - `urban`, whether the road is in an urban area (derived from speed limit).
#
# Unfortunately, not all fields are actually available in the dataset.

road_features = set(
    [
        "flatHill",
        "junctionType",
        "NumberOfLanes",
        "roadCharacter",
        "roadCurvature",
        "roadLane",
        "roadMarkings",
        "roadSurface",
        "speedLimit",
        "streetLight",
        "urban",
    ]
)
missing_features = road_features - set(dset.columns)
road_features -= missing_features
print("The following features are not found in the dataset:", missing_features)

# +
fig, axes = plt.subplots(3, 3)

for ax, feat in it.zip_longest(axes.flat, sorted(road_features)):
    if feat is None:
        ax.axis("off")
        continue
    counts = dset[feat].value_counts(dropna=False)
    counts.plot.bar(ylabel="# crashes", figsize=(10, 5), ax=ax, title=feat)
    ax.set(yscale="log")

fig.set_size_inches(15, 12)
fig.tight_layout()
# -

# **TODO** comment on which features to keep in the end

# ## Environmental features
#
# The environmental features are weather and sunhsine:
#
# - `light`, light at the time and place of the crash, **note: this can also be
#   used as a time indication**,
# - `weatherA` and `weatherB`,  weather at the crash time/place.

# +
env_features = ["light", "weatherA", "weatherB"]

fig, axes = plt.subplots(1, 3)
for ax, feat in zip(axes.flat, env_features):
    counts = dset[feat].value_counts(dropna=False)
    counts.plot.bar(ylabel="# crashes", figsize=(10, 5), ax=ax, title=feat)
    ax.set(yscale="log")
fig.set_size_inches(13, 4)
fig.tight_layout()
# -

# ## Next steps
#
# We have checked the spatial, temporal, road and environmental features related
# to the accidents.
#
# If these features inform us in which conditions there are more accidents
# relatively, we will need additional baseline information if we want to create
# a predictive model.
#
# For the road features we could use another [NZTA dataset](https://opendata-nzta.opendata.arcgis.com/datasets/NZTA::national-road-centreline-road-controlling-authority-data)
# that brings more information about the road type and traffic. But then we need
# to attribute each crash to a road.
#
# Another option would be to regrid the data, and for each cell containing at
# least one crash event we associate road features from the crash events. With
# this option, we don't make any prediction for cells of the grid where we don't
# have information about.
#
# For the environmental information, we need weather information for all days in
# a year:
#
# - for example from a climate reanalysis, e.g. [ERA 5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview),
# - or we can use the [station data from NIWA](https://cliflo.niwa.co.nz/),
# - or some simple [climate summaries](https://niwa.co.nz/education-and-training/schools/resources/climate/summary).
#
# The prediction task can be formulated in different ways:
#
# - If we exclude the environmental features, we can cast the problem as a
#   regression with count data, for each year & location.
# - Otherwise we need to decide on a time bin (e.g. a day) and predict the
#   probability of at least one crash for each time bin & location.
#   In this case we need to generate negative samples for every time of the
#   year & location when no crash happened.
# - Finally we can also try to predict the severity of the accidents, which is a
#   slightly different problem.
