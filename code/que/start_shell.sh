#!/bin/bash

ENV_NAME="wlasl"
# ENV_NAME="wlasl_cpu"

# Source conda
CONDA_PATH="/home/luke/miniconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"

# Activate environment
conda activate "$ENV_NAME"

# CD to project and run shell
cd /home/luke/Code/SLR/code
exec python -m que.shell