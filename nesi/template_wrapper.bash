#!/usr/bin/env bash

# This script is used in ~/.local/share/jupyter/kernels/<kernel_name>/kernel.json
# to load environment modules before starting the kernel.

# clean list of modules on Mahuika
if hostname | grep -vq "maui"; then
    module purge
fi

# load required modules here
module load slurm

# load conda & CUDA modules on Mahuika or Maui
if hostname | grep -q "maui"; then
    CONDA_MODULE="Anaconda3/2020.02-GCC-7.1.0"
else
    CONDA_MODULE="Miniconda3/4.8.2"
fi
module load "$CONDA_MODULE"

# ensure user packages are not used by conda
export PYTHONNOUSERSITE=1

# activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate  # enforce base environment to be unloaded
conda activate ##CONDA_VENV_PATH##

# run the kernel
exec python $@
