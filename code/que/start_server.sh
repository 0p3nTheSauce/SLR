#!/bin/bash
set -e  # Exit on error

# Log start
echo "$(date): Starting ML training service"

# Source conda (adjust path to your conda installation)
CONDA_PATH="/home/luke/miniconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"


#for debugging purposes,
ENV_NAME="wlasl"
# ENV_NAME="wlasl_cpu"

# Activate environment
conda activate "$ENV_NAME"

# Verify environment is activated
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

# Log Python and package info
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Change to que directory
cd /home/luke/Code/SLR/code

# Run the server using the module syntax
exec python -m que.server