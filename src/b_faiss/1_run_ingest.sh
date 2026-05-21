#!/bin/bash
#SBATCH --account=project_xxx
#SBATCH --output=./log/ingest/%j/output.log
#SBATCH --error=./log/ingest/%j/error.log
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=2:00:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7

VENV_DIR="/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/venv"
source $VENV_DIR/bin/activate

export HF_HUB_CACHE=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub/

python -c "from langchain_text_splitters import RecursiveCharacterTextSplitter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Langchain TextSplitter not found. Installing..."
    pip install langchain-text-splitters
else
    echo "Langchain TextSplitter already installed."
fi

MODEL=$1
JSON_FILE_PATH=/scratch/${SLURM_JOB_ACCOUNT}/${SLURM_JOB_USER}/$2
FAISS_INDEX_PATH=/scratch/${SLURM_JOB_ACCOUNT}/${SLURM_JOB_USER}/$3
CHUNK_SIZE=$4
OVERLAP=$5
BATCH_SIZE=$6

srun torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    src/b_faiss/ingest_faiss.py \
    --model $MODEL \
    --filepath $JSON_FILE_PATH \
    --faiss_index_path $FAISS_INDEX_PATH \
    --chunk_size $CHUNK_SIZE \
    --overlap $OVERLAP \
    --batch_size $BATCH_SIZE
