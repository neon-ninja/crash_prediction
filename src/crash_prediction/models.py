import typing as T
import pickle
import itertools as it
from enum import Enum
from pathlib import Path

import defopt
import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import dask_ml.model_selection as dcv


def columns_transform():
    return make_column_transformer(
        (
            StandardScaler(),
            make_column_selector("^(?!crashYear)", dtype_include=np.number),
        ),
        (
            OneHotEncoder(handle_unknown="ignore"),
            make_column_selector(dtype_include=object),
        ),
    )


def fit_dummy(X, y, n_iter):
    """Fit a dummy estimator"""
    model = DummyClassifier(strategy="prior")
    model.fit(X, y)
    return model


def fit_linear(X, y, n_iter):
    """Fit a logistic regression model"""
    model = LogisticRegression(max_iter=500, penalty="elasticnet", solver="saga")
    model = make_pipeline(columns_transform(), model)

    param_space = {
        "logisticregression__l1_ratio": st.uniform(0, 1),
        "logisticregression__C": st.loguniform(1e-4, 1e4),
    }
    model = dcv.RandomizedSearchCV(
        model, param_space, scoring="neg_log_loss", n_iter=n_iter, random_state=42, cv=5
    )

    model.fit(X, y)
    return model


def fit_mlp(X, y, n_iter):
    """Fit a simple multi-layer perceptron model"""
    model = MLPClassifier(random_state=42, early_stopping=True)
    model = make_pipeline(columns_transform(), model)

    layers_options = [
        [n_units] * n_layers
        for n_units, n_layers in it.product([32, 64, 128, 256, 512], [1, 2])
    ]
    param_space = {
        "mlpclassifier__hidden_layer_sizes": layers_options,
        "mlpclassifier__alpha": st.loguniform(1e-5, 1e-2),
        "mlpclassifier__learning_rate_init": st.loguniform(1e-4, 1e-1),
    }
    model = dcv.RandomizedSearchCV(
        model, param_space, scoring="neg_log_loss", n_iter=n_iter, random_state=42, cv=5
    )

    model.fit(X, y)
    return model


def loguniform_int(a, b):
    """Create a discrete random variable following a log-uniform distribution"""
    xs = np.arange(a, b + 1)
    probs = 1 / (xs * np.log(b / a))
    probs /= probs.sum()
    return st.rv_discrete(a=a, b=b, values=[xs, probs], name="loguniform_int")


def fit_knn(X, y, n_iter):
    """Fit a KNN model on geographical coordinates only"""
    columns_tf = make_column_transformer(("passthrough", ["X", "Y"]))
    model = make_pipeline(columns_tf, KNeighborsClassifier())

    param_space = {
        "kneighborsclassifier__n_neighbors": loguniform_int(1, 500),
        "kneighborsclassifier__weights": ["uniform", "distance"],
    }
    model = dcv.RandomizedSearchCV(
        model, param_space, scoring="neg_log_loss", n_iter=n_iter, random_state=42, cv=5
    )

    model.fit(X, y)
    return model


def fit_gbdt(X, y, n_iter):
    """Fit a gradient boosted decision trees model"""
    model = LGBMClassifier(n_estimators=2000, random_state=42)
    model = make_pipeline(columns_transform(), model)

    param_space = {
        "lgbmclassifier__min_data_in_leaf": loguniform_int(5, 500),
        "lgbmclassifier__num_leaves": loguniform_int(31, 500),
        "lgbmclassifier__reg_alpha": st.loguniform(1e-10, 1.0),
        "lgbmclassifier__reg_lambda": st.loguniform(1e-10, 1.0),
        "lgbmclassifier__learning_rate": st.loguniform(1e-4, 1e-1),
    }
    model = dcv.RandomizedSearchCV(
        model, param_space, scoring="neg_log_loss", n_iter=n_iter, random_state=42, cv=5
    )

    model.fit(X, y)
    return model


def slurm_cluster(n_workers, cores_per_worker, mem_per_worker, walltime, dask_folder):
    """helper function to start a Dask Slurm-based cluster

    :param n_workers: maximum number of workers to use
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
    cluster.adapt(minimum=1, maximum=n_workers)

    client = Client(cluster)
    return client


ModelType = Enum("ModelType", "dummy linear mlp knn gbdt")


def fit(
    dset: Path,
    output_file: Path,
    *,
    model_type: ModelType = ModelType.linear,
    n_iter: int = 50,
    n_workers: int = 1,
    cores_per_worker: int = 4,
    dask_folder: Path = Path.cwd() / "dask",
    mem_per_worker: str = "2GB",
    walltime: str = "0-00:30",
    use_slurm: bool = False,
) -> BaseEstimator:
    """Fit a model

    :param dset: CAS dataset
    :param output_file: output .pickle file
    :param model_type: type of model to use
    :param n_iter: budget for hyper-parameters optimization
    :param n_workers: number of workers to use, maximum number for Slurm backend
    :param cores_per_worker: number of cores per worker
    :param dask_folder: folder to keep workers temporary data
    :param mem_per_worker: maximum of RAM for workers, only for Slurm backend
    :param walltime: maximum time for workers, only for Slurm backend
    :param use_slurm: use Slurm backend for the Dask cluster
    :returns: fitted model
    """
    dset = pd.read_csv(dset)
    X = dset[dset.fold == "train"].drop(columns="fold")
    y = X.pop("injuryCrash")

    # find function to fit the model in the global namespace
    model_func = globals()["fit_" + model_type.name]

    # start a Dask cluster, local by default, use a configuration file for Slurm
    if use_slurm:
        client = slurm_cluster(
            n_workers=n_workers,
            cores_per_worker=cores_per_worker,
            mem_per_worker=mem_per_worker,
            walltime=walltime,
            dask_folder=dask_folder,
        )
    else:
        client = Client(
            n_workers=n_workers,
            threads_per_worker=cores_per_worker,
            local_directory=dask_folder,
        )

    client.wait_for_workers(1)
    model = model_func(X, y, n_iter=n_iter)

    with output_file.open("wb") as fd:
        pickle.dump(model, fd)


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

    X = dset.drop(columns=["injuryCrash", "fold"])

    y_prob = model.predict_proba(X)
    y_prob = pd.Series(y_prob[:, 1], name="crashInjuryProb")

    if output_file is not None:
        y_prob.to_csv(output_file, index=False)

    return y_prob


def main():
    """wrapper function to create a CLI tool"""
    defopt.run(
        [fit, predict],
        parsers={T.Union[pd.DataFrame, Path]: Path, T.Union[BaseEstimator, Path]: Path},
    )
