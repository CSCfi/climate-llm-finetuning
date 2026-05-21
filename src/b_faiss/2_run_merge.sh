#!/bin/bash
#SBATCH --account=project_xxx
#SBATCH --output=./log/merge/%j/output.log
#SBATCH --error=./log/merge/%j/error.log
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=0:15:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7

FAISS_INDEX_PATH=/scratch/${SLURM_JOB_ACCOUNT}/${SLURM_JOB_USER}/$1

srun python src/b_faiss/merge_faiss.py --faiss_index_path $FAISS_INDEX_PATH
