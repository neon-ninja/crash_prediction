from pathlib import Path

import requests
import defopt

CASDATA_URL = (
    "https://opendata.arcgis.com/datasets/8d684f1841fa4dbea6afaefc8a1ba0fc_0.csv"
)


def download(output_file: Path, url: str = CASDATA_URL):
    """Download the Crash Analysis System dataset and save it in a .csv file

    Note that parents folders of the output file are created if needed.

    :param output_file: output .csv file path
    :param url: dataset URL
    """
    dset_web = requests.get(url)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with output_file.open("wb") as fd:
        fd.write(dset_web.content)


def main():
    """wrapper function to create a CLI tool"""
    defopt.run([download])
