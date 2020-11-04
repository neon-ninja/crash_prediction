import pandas as pd
import hvplot.pandas
import hvplot  # NOQA

# download page: https://opendata-nzta.opendata.arcgis.com/datasets/crash-analysis-system-cas-data-1
# fields description: https://opendata-nzta.opendata.arcgis.com/pages/cas-data-field-descriptions

dset = pd.read_csv("../../data/Crash_Analysis_System__CAS__Data.csv")

# New Zealand Transverse Mercator 2000 (NZTM2000) projection https://epsg.io/2193

hv_map = dset.hvplot.points(
    "X",
    "Y",
    datashade=True,
    geo=True,
    tiles="OSM",
    crs=2193,
    frame_width=600,
)

hvplot.show(hv_map)
