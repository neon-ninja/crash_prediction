# # Exploration of CAS (Crash analysis system) data

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sb
import contextily as cx

from crash_prediction import cas_data

# set seaborn default style
sb.set()

# ## Data retrieval

# First we need to retrieve the dataset from the [Open Data portal](https://opendata-nzta.opendata.arcgis.com/datasets/crash-analysis-system-cas-data-1).
# Multiple file formats are available (csv, kml, geojson, ...), the most compact
# being the .csv one.

dset_path = Path("..") / "data" / "cas_dataset.csv"
if not dset_path.exists():
    dset_path.parent.mkdir(parents=True, exist_ok=True)
    cas_data.download(dset_path)

# Next we load the data and have a quick look to check if there no obvious
# loading error.

dset = pd.read_csv(dset_path)
dset

# The dataset contains 72 columns, describing various aspects of the recorded
# car crashes. The full description of the fields is available online, see
# https://opendata-nzta.opendata.arcgis.com/pages/cas-data-field-descriptions.

dset.columns

# Note that `X` and `Y` are geographical coordinates using the WGS84 coordinate
# system (see [EPSG:4326](https://epsg.io/4326)).

# ## Spatial features
#
# First, we will look at the location of the crashes. More accidents happen in
# densier areas and it would be good to compare with population density.
#
# *Note: We removed Chatham island data here to ease plotting.*


def plot_hexmap(dset, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    hb = ax.hexbin(
        dset["X"], dset["Y"], gridsize=500, cmap="BuPu", mincnt=1, bins="log"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cx.add_basemap(ax, crs=4326, source=cx.providers.CartoDB.Positron)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(hb, cax=cax)

    return ax


plot_hexmap(dset[dset.X > 0])

# In dense aread, like in Auckland, there are enough crashes events to map the
# local road network.

dset_auckland = dset[dset["X"].between(174.7, 174.9) & dset["Y"].between(-37, -36.8)]
plot_hexmap(dset_auckland)

# At a coarser level, there is also the region information.

region_perc = dset["region"].value_counts(normalize=True)
ax = region_perc.plot.bar(ylabel="fraction of crashes", figsize=(10, 5))
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

print(
    f"The top 4 regions account for {region_perc.nlargest(4).sum() * 100:0.1f}% "
    "of the crashes."
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
_ = year_counts.plot.bar(ylabel="# crashes", figsize=(10, 5))

year_region_counts = (
    dset.groupby(["crashYear", "region"]).size().reset_index(name="# crashes")
)
_, ax = plt.subplots(figsize=(10, 5))
sb.pointplot(data=year_region_counts, x="crashYear", y="# crashes", hue="region", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
_ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

# We can also explore the spatio-temporal patterns too. Here we focus on
# Auckland (excluding 2020).

grid = sb.FacetGrid(
    dset_auckland[dset_auckland.crashYear < 2020], col="crashYear", col_wrap=5
)
grid.map(plt.hexbin, "X", "Y", gridsize=500, cmap="BuPu", mincnt=1, bins="log")

# The other temporal attribute is the holiday. Christmas is the holiday period
# with most of the accidents. How the period is computed is not clear, so the
# larger amount of accident could be partly due to the time extent. Easter,
# Queens Birthday and Labour weekend are 3 to 4 days periods. Christmas & New Year
# is probably 1 to 2 weeks period.

holiday_counts = dset["holiday"].fillna("Normal day").value_counts()
ax = holiday_counts.plot.bar(ylabel="# crashes", rot=0, figsize=(10, 5))
_ = ax.set(yscale="log")

# ## Road Features
#
# From the dataset fields description, the following features seem specific to
# the type of road:
#
# - `crashSHDescription`, whether the crash happened on a state highway,
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
# - `streetLight`, street lighting at the time of the crash (this is also
#   a sort of **temporal information**),
# - `urban`, whether the road is in an urban area (derived from speed limit).
#
# Unfortunately, not all fields are actually available in the dataset.

road_features = set(
    [
        "crashSHDescription",
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

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, feat in zip(axes.flat, sorted(road_features)):
    counts = dset[feat].value_counts(dropna=False)
    counts.plot.bar(ylabel="# crashes", title=feat, ax=ax)
    ax.set(yscale="log")
fig.tight_layout()

# The `urban` feature is derived from `speedLimit`, so we can probably remove it.

# ## Environmental features
#
# The environmental features are weather and sunhsine:
#
# - `light`, light at the time and place of the crash (this is also a sort of
#   **temporal information**),
# - `weatherA` and `weatherB`,  weather at the crash time/place.

env_features = ["light", "weatherA", "weatherB"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, feat in zip(axes.flat, env_features):
    counts = dset[feat].value_counts(dropna=False)
    counts.plot.bar(ylabel="# crashes", title=feat, ax=ax)
    ax.set(yscale="log")
fig.tight_layout()

# ## Possible next steps
#
# We have checked the spatial, temporal, road and environmental features related
# to the accidents.
#
# If these features inform us in which conditions there are more accidents
# relatively, we will need additional baseline information if we want to create
# a predictive model.
#
# For the road features we could use a [LINZ dataset](https://data.linz.govt.nz/layer/50329-nz-road-centrelines-topo-150k/)
# or another [NZTA dataset](https://opendata-nzta.opendata.arcgis.com/datasets/NZTA::national-road-centreline-road-controlling-authority-data)
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
# 1. exclude weather & holiday features, and fit a regression model with count
#    data using year & location features (and accounting for traffic volume to
#    compare the number of crashes per car on the road),
#
# 2. group data by location, time, weather type (e.g. rain vs. no rain), and
#    perform a binomial regression using the total number of days in each
#    category (e.g. number of rain days for a particular location & year),
#
# 3. predict crash severity from the whole dataset, assuming the non-severe
#    crashes are a good proxy for normal conditions (weather, holidays, etc.).

# ---
# ## Original computing environment

# !date -R

# !uname -a

# !pip freeze
