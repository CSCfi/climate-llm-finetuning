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
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif

singularity run $SIF bash -c "python -m venv --system-site-packages /scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/marimo_lumi_venv && source /scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/marimo_lumi_venv/bin/activate && pip install marimo==0.23.0 PyMuPDF==1.27.2.3 lxml==5.4.0"

export PYTHONPATH=$PYTHONPATH:/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/marimo_lumi_venv/lib/python3.12/site-packages

singularity run $SIF bash -c "source /scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/marimo_lumi_venv/bin/activate && python -c 'from src.a_data.data_gather import sbatch_main; sbatch_main()'"