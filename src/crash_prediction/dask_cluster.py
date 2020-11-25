import sys
import signal
from pathlib import Path

import defopt

import dask
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster


def slurm_cluster(n_workers, cores_per_worker, mem_per_worker, walltime, dask_folder):
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


def dask_cluster(
    *,
    n_workers: int = 1,
    threads_per_worker: int = 4,
    dask_folder: Path = Path.cwd() / "dask",
    use_slurm: bool = False,
    walltime: str = "0-00:30",
    mem_per_worker: str = "2GB",
):
    """Start a Dask cluster

    By default, a local cluster is created. Run forever until interrupted.

    :param n_workers: number of workers to use
    :param threads_per_worker: number of cores per worker
    :param dask_folder: folder to keep workers temporary data
    :param use_slurm: use Slurm as backend
    :param walltime: maximum time for workers (for Slurm backend only)
    :param mem_per_worker: maximum of RAM for workers (for Slurm backend only)
    """
    if use_slurm:
        cluster = slurm_cluster(
            n_workers, threads_per_worker, mem_per_worker, walltime, dask_folder
        )
    else:
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            local_directory=dask_folder,
        )

    def signal_handler(sig, frame):
        cluster.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.pause()


def main():
    """wrapper function to create a CLI tool"""
    defopt.run(dask_cluster)
