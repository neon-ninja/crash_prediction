import typing as T
from pathlib import Path

import requests
import defopt
import pandas as pd

CAS_DATA_URL = (
    "https://opendata.arcgis.com/datasets/8d684f1841fa4dbea6afaefc8a1ba0fc_0.csv"
)


def download(output_file: Path, url: str = CAS_DATA_URL):
    """Download the Crash Analysis System dataset and save it in a .csv file

    Note that parents folders of the output file are created if needed.

    :param output_file: output .csv file path
    :param url: dataset URL
    """
    dset_web = requests.get(url)
    with output_file.open("wb") as fd:
        fd.write(dset_web.content)


def prepare(
    input_data: T.Union[pd.DataFrame, Path],
    *,
    output_file: T.Optional[Path] = None,
    test_size: float = 0.2
) -> pd.DataFrame:
    """Prepare CAS dataset, cleaning unknown data and selecting features

    Note that `NumberOfLanes` may contains NaN values in the returned DataFrame.

    :param input_data: input CAS dataset
    :param output_file: output .csv file
    :param test_size: size of the test dataset, as a fraction of the full dataset
    :returns: preprocessed data
    """

    if isinstance(input_data, Path):
        input_data = pd.read_csv(input_data)

    # sub-select relevant features
    features = [
        # spatial features
        "X",
        "Y",
        "region",
        # temporal features
        "crashYear",
        "holiday",
        # road features
        "crashSHDescription",
        "flatHill",
        "NumberOfLanes",
        "roadCharacter",
        "roadLane",
        "roadSurface",
        "speedLimit",
        "streetLight",
        # environmental features
        "light",
        "weatherA",
        "weatherB",
        # target variable
        "crashSeverity",
    ]
    input_data = input_data[features].copy()

    # remove NaN values where it's not obvious what they stand for
    input_data.dropna(subset=["crashSHDescription", "region"], inplace=True)

    # NaNs in "holiday" stands for regular days
    input_data["holiday"].fillna("Normal day", inplace=True)

    # NaNs in "speedLimit" stands for LSZ zone
    input_data["LSZ"] = input_data["speedLimit"].isna()
    input_data["speedLimit"].fillna(100, inplace=True)

    # homogenize unknown values representation
    input_data.replace("Null", "Unknown", inplace=True)

    # sort by year and flag the last entries as test data
    input_data.sort_values(by="crashYear", ascending=True, inplace=True)

    test_idx = int(len(input_data) * test_size)
    input_data["fold"] = "train"
    input_data.loc[input_data.index[-test_idx:], "fold"] = "test"

    # regroup serious and fatal crashes in the same class
    input_data["crashSeverity"].replace("Fatal Crash", "Serious Crash", inplace=True)

    if output_file is not None:
        input_data.to_csv(output_file, index=False)

    return input_data


def main():
    """wrapper function to create a CLI tool"""
    defopt.run([download, prepare], parsers={T.Union[pd.DataFrame, Path]: Path})
