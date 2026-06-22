#!/bin/bash
#SBATCH --account=project_xxx
#SBATCH --output=./log/data/%j/output.log
#SBATCH --error=./log/data/%j/error.log
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=5:00:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7

VENV_DIR="/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/venv"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    echo "Activating venv at $VENV_DIR"
    source $VENV_DIR/bin/activate
else
    echo "Creating new venv at $VENV_DIR"
    python -m venv $VENV_DIR --system-site-packages

    echo "Activating created venv at $VENV_DIR"
    source $VENV_DIR/bin/activate

    pip list

    which python

    python -m pip install marimo==0.20.4 pymupdf
fi

python -c "from src.a_data.data_gather import sbatch_main; sbatch_main()"
