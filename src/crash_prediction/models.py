import typing as T
import pickle
from pathlib import Path

import defopt
import numpy as np
import pandas as pd
import scipy.stats as st
import yaml

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import dask_ml.model_selection as dcv


def slurm_cluster(n_workers, cores_per_worker, mem_per_worker, walltime, dask_folder):
    """start a Dask Slurm-based cluster

    :param n_workers: number of workers to use
    :param cores_per_worker: number of cores per worker
    :param mem_per_worker: maximum of RAM for workers
    :param walltime: maximum time for workers
    :param dask_folder: folder to keep workers temporary data
    """
    dask.config.set(
        {
            "distributed.worker.memory.target": False,  # avoid spilling to disk
            "distributed.worker.memory.spill": False,  # avoid spilling to disk
        }
    )
    cluster = SLURMCluster(
        cores=cores_per_worker,
        processes=1,
        memory=mem_per_worker,
        walltime=walltime,
        log_directory=dask_folder / "logs",  # folder for SLURM logs for each worker
        local_directory=dask_folder,  # folder for workers data
    )
    cluster.scale(n=n_workers)
    return cluster


def split_data(dset, fold):
    X = dset[dset.fold == fold].drop(columns="fold")
    y = X.pop("injuryCrash")
    return X, y


def columns_transform():
    # TODO include year?
    return make_column_transformer(
        ("drop", "crashYear"),
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (
            OneHotEncoder(handle_unknown="ignore"),
            make_column_selector(dtype_include=object),
        ),
    )


def fit_linear(
    dset: T.Union[pd.DataFrame, Path],
    *,
    output_file: T.Optional[Path] = None,
    fold: str = "train",
    verbose: bool = False,
    jobs: int = 1,
) -> BaseEstimator:
    """Fit a logistic regression model

    :param dset: CAS dataset
    :param output_file: output .pickle file
    :param fold: fold used for training
    :param verbose: verbose mode
    :param jobs: number of jobs to use, -1 means all processors
    :returns: fitted model
    """
    if isinstance(dset, Path):
        dset = pd.read_csv(dset)

    X, y = split_data(dset, fold)

    model = LogisticRegressionCV(
        max_iter=500, scoring="neg_log_loss", n_jobs=jobs, verbose=verbose
    )
    model = make_pipeline(columns_transform(), model)

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
    n_iter: int = 10,
    jobs: int = 1,
    dask_folder: Path = Path.cwd() / "dask",
    slurm_config: T.Optional[Path] = None,
) -> BaseEstimator:
    """Fit a multi-layer perceptron model

    :param dset: CAS dataset
    :param output_file: output .pickle file
    :param fold: fold used for training
    :param verbose: verbose mode
    :param n_iter: number of random configurations to test
    :param jobs: number of jobs to use, ignored if a Dask cluster is used
    :param dask_folder: folder to keep workers temporary data
    :param slurm_config: Dask Slurm-based cluster .yaml configuration file
    :returns: fitted model
    """
    if isinstance(dset, Path):
        dset = pd.read_csv(dset)

    X, y = split_data(dset, fold)

    model = MLPClassifier(random_state=42, early_stopping=True)
    model = make_pipeline(columns_transform(), model)

    param_space = {
        "mlpclassifier__alpha": st.loguniform(1e-5, 1e-2),
        "mlpclassifier__learning_rate_init": st.loguniform(1e-4, 1e-1),
    }
    model = dcv.RandomizedSearchCV(
        model, param_space, scoring="neg_log_loss", n_iter=n_iter, random_state=42
    )

    if slurm_config is None:
        client = Client(n_workers=jobs, local_directory=dask_folder)
    else:
        slurm_kwargs = yaml.safe_load(slurm_config.read_text())
        cluster = slurm_cluster(**slurm_kwargs, dask_folder=dask_folder)
        client = Client(cluster)

    client.wait_for_workers(1)
    model.fit(X, y)

    if output_file is not None:
        with output_file.open("wb") as fd:
            pickle.dump(model, fd)

    return model


def fit_knn(
    dset: T.Union[pd.DataFrame, Path],
    *,
    output_file: T.Optional[Path] = None,
    fold: str = "train",
    verbose: bool = False,
    jobs: int = 1,
) -> BaseEstimator:
    """Fit a KNN model

    The number of neighbors is selected via cross-validation.

    :param dset: CAS dataset
    :param output_file: output .pickle file
    :param fold: fold used for training
    :param verbose: verbose mode
    :param jobs: number of jobs to use, -1 means all processors
    :returns: fitted model
    """
    if isinstance(dset, Path):
        dset = pd.read_csv(dset)

    X, y = split_data(dset, fold)

    columns_tf = make_column_transformer(("passthrough", ["X", "Y"]))
    model = make_pipeline(columns_tf, KNeighborsClassifier())

    param_grid = {
        "kneighborsclassifier__n_neighbors": [1, 5, 10, 20, 50, 100, 200, 500]
    }
    model = GridSearchCV(
        model, param_grid, scoring="neg_log_loss", verbose=verbose, n_jobs=jobs
    )

    model.fit(X, y)

    if output_file is not None:
        with output_file.open("wb") as fd:
            pickle.dump(model, fd)

    return model


def fit_rf(
    dset: T.Union[pd.DataFrame, Path],
    *,
    output_file: T.Optional[Path] = None,
    fold: str = "train",
    verbose: bool = False,
    jobs: int = 1,
    dask_folder: Path = Path.cwd() / "dask",
    slurm_config: T.Optional[Path] = None,
) -> BaseEstimator:
    """Fit a random forest model

    :param dset: CAS dataset
    :param output_file: output .pickle file
    :param fold: fold used for training
    :param verbose: verbose mode
    :param jobs: number of jobs to use, ignored if a Dask cluster is used
    :param dask_folder: folder to keep workers temporary data
    :param slurm_config: Dask Slurm-based cluster .yaml configuration file
    :returns: fitted model
    """
    if isinstance(dset, Path):
        dset = pd.read_csv(dset)

    X, y = split_data(dset, fold)

    model = RandomForestClassifier(random_state=42, verbose=verbose)
    model = make_pipeline(columns_transform(), model)

    if slurm_config is None:
        client = Client(n_workers=jobs, local_directory=dask_folder)
    else:
        slurm_kwargs = yaml.safe_load(slurm_config.read_text())
        cluster = slurm_cluster(**slurm_kwargs, dask_folder=dask_folder)
        client = Client(cluster)

    client.wait_for_workers(1)
    with joblib.parallel_backend("dask", scatter=[X, y]):
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
) -> pd.Series:
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
    y_prob = pd.Series(y_prob[:, 1], name="crashInjuryProb")

    if output_file is not None:
        y_prob.to_csv(output_file, index=False)

    return y_prob


def main():
    """wrapper function to create a CLI tool"""
    defopt.run(
        [fit_linear, fit_mlp, fit_knn, fit_rf, predict],
        parsers={T.Union[pd.DataFrame, Path]: Path, T.Union[BaseEstimator, Path]: Path},
    )
