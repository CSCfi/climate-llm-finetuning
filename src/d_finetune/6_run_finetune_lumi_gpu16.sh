#!/bin/bash
#SBATCH --account=project_xxx
#SBATCH --partition=standard-g
#SBATCH --output=./log/finetuning_accelerate/%j/output.log
#SBATCH --error=./log/finetuning_accelerate/%j/error.log
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=8

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif

# This will store all the Hugging Face cache such as downloaded models
# and datasets in the project's scratch folder
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub/
mkdir -p $HF_HOME

export HF_TOKEN_PATH=~/.cache/huggingface/token

MODEL=$1
OUTPUT=$2
DATA=$3
BATCH=$4
WORKERS=$5

# Disable internal parallelism of huggingface's tokenizer since we
# want to retain direct control of parallelism options.
export TOKENIZERS_PARALLELISM=false

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}" # set port based on SLURM_JOB_ID to avoid conflicts

export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

set -xv  # print the command so that we can verify setting arguments correctly from the logs

srun singularity run $SIF python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --node_rank $SLURM_PROCID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    src/d_finetune/train_accelerate.py $* \
    --model $MODEL \
    --json-datapath /scratch/${SLURM_JOB_ACCOUNT}/${USER}/"$DATA" \
    --output-path /scratch/${SLURM_JOB_ACCOUNT}/${USER}/"$OUTPUT" \
    --num-workers $WORKERS \
    --batch_size $BATCH \
    --peft \
    # --4bit \