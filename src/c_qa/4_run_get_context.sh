#!/bin/bash
#SBATCH --account=project_xxx
#SBATCH --partition=dev-g
#SBATCH --output=./log/get_context/%j/output.log
#SBATCH --error=./log/get_context/%j/error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=1:00:00
#SBATCH --nodes=1

#export OMP_NUM_THREADS=1  # Set the number of OpenMP threads

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7

VENV_DIR="/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/venv"
echo "Activating venv at $VENV_DIR"
source $VENV_DIR/bin/activate

export HF_HUB_CACHE=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub/

MODEL=$1
JSON_FILE_PATH=/scratch/${SLURM_JOB_ACCOUNT}/${SLURM_JOB_USER}/$2
FAISS_FILE_PATH=/scratch/${SLURM_JOB_ACCOUNT}/${SLURM_JOB_USER}/$3
SUFFIX=$4

srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    src/c_qa/get_context.py \
    --model $MODEL \
    --jsonpath $JSON_FILE_PATH \
    --faisspath $FAISS_FILE_PATH \
    --suffix $SUFFIX
