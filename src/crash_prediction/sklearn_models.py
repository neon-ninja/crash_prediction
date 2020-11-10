import typing as T
import pickle
from pathlib import Path

import defopt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def split_data(dset):
    X = dset.drop(columns=["crashSeverity", "fold", "NumberOfLanes"])
    y = dset["crashSeverity"]
    return X, y


def fit_linear(
    dset: T.Union[pd.DataFrame, Path],
    *,
    output_file: T.Optional[Path] = None,
    fold: str = "train",
    verbose: bool = False,
) -> BaseEstimator:
    """Fit a logistic regression model

    :param dset: CAS dataset
    :param output_file: output .pickle file
    :param fold: fold used for training
    :param verbose: verbose mode
    :returns: fitted model
    """
    if isinstance(dset, Path):
        dset = pd.read_csv(dset)

    X, y = split_data(dset[dset.fold == fold])

    columns_tf = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (
            OneHotEncoder(handle_unknown="ignore"),
            make_column_selector(dtype_include=object),
        ),
    )
    model = make_pipeline(columns_tf, LogisticRegression(verbose=verbose))
    model.fit(X, y)

    if output_file is not None:
        with output_file.open("wb") as fd:
            pickle.dump(model, fd)

    return model


def fit_mlp(
    dset: T.Union[pd.DataFrame, Path],
    *,
    output_file: T.Optional[Path] = None,
    fold: str = "train",
    verbose: bool = False,
) -> BaseEstimator:
    """Fit a multi-layer perceptron model

    :param dset: CAS dataset
    :param output_file: output .pickle file
    :param fold: fold used for training
    :param verbose: verbose mode
    :returns: fitted model
    """
    if isinstance(dset, Path):
        dset = pd.read_csv(dset)

    X, y = split_data(dset[dset.fold == fold])

    columns_tf = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (
            OneHotEncoder(handle_unknown="ignore"),
            make_column_selector(dtype_include=object),
        ),
    )
    model = make_pipeline(columns_tf, MLPClassifier(random_seed=42, verbose=verbose))
    model.fit(X, y)

    if output_file is not None:
        with output_file.open("wb") as fd:
            pickle.dump(model, fd)

    return model


def predict(
    dset: T.Union[pd.DataFrame, Path],
    model: T.Union[BaseEstimator, Path],
    *,
    output_file: T.Optional[Path] = None,
) -> pd.DataFrame:
    """Make predictions from a fitted model

    :param dset: CAS dataset
    :param model: trained model
    :param output_file: output .csv file
    :returns: predictions
    """
    if isinstance(dset, Path):
        dset = pd.read_csv(dset)
    if isinstance(model, Path):
        with model.open("rb") as fd:
            model = pickle.load(fd)

    X, _ = split_data(dset)
    y_prob = model.predict_proba(X)
    y_prob = pd.DataFrame(y_prob, columns=model.steps[1][1].classes_)

    if output_file is not None:
        y_prob.to_csv(output_file, index=False)

    return y_prob


def main():
    """wrapper function to create a CLI tool"""
    defopt.run(
        [fit_linear, fit_mlp, predict],
        parsers={T.Union[pd.DataFrame, Path]: Path, T.Union[BaseEstimator, Path]: Path},
    )
