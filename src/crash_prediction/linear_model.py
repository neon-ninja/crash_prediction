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


def split_data(dset):
    X = dset.drop(columns=["crashSeverity", "fold", "NumberOfLanes"])
    y = dset["crashSeverity"]
    return X, y


def fit_(dset, fold, verbose):
    X, y = split_data(dset[dset.fold == fold])

    columns_tf = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(), make_column_selector(dtype_include=object)),
    )
    model = make_pipeline(columns_tf, LogisticRegression(verbose=verbose))
    model.fit(X, y)

    return model


def fit(
    dataset_path: Path, model_path: Path, *, fold: str = "train", verbose: bool = False
):
    """Fit a logistic regression model

    :param dataset_path: CAS dataset .csv file path
    :param model_path: output model .pickle file path
    :param fold: fold used for training
    :param verbose: verbose mode
    """
    dset = pd.read_csv(dataset_path)
    model = fit_(dset, fold, verbose)
    with model_path.open("wb") as fd:
        pickle.dump(model, fd)


def predict_(dset, model):
    X, _ = split_data(dset)
    y_prob = model.predict_proba(X)
    return pd.DataFrame(y_prob, columns=model.steps[1][1].classes_)


def predict(dataset_path: Path, model_path: Path, preds_path: Path):
    """Predict crash severity from a fitted scikit-learn model

    :param dataset_path: CAS dataset .csv file path
    :param model_path: model .pickle file path
    :param preds_path: output predictions .csv file path
    """
    dset = pd.read_csv(dataset_path)
    with model_path.open("rb") as fd:
        model = pickle.load(fd)
    y_prob = predict_(dset, model)
    y_prob.to_csv(preds_path)


def main():
    """wrapper function to create a CLI tool"""
    defopt.run([fit, predict])