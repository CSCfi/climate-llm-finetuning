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

# Problem with located when trying to run finetuning in lumi singularity container: transformers version should be > 5, otherwise ImportError: cannot import name 'EmbeddingParallel' from 'transformers.integrations.tensor_parallel'
singularity run $SIF bash -c "python -m venv --system-site-packages ./ft_venv && source ./ft_venv/bin/activate && pip install -U transformers==5.5.4"

export PYTHONPATH=$PYTHONPATH:./ft_venv/lib/python3.12/site-packages

export HF_TOKEN_PATH=~/.cache/huggingface/token

ACCELERATE_CONFIG=$1  # first argument must be a1ccelerate config to use
if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "ERROR: first argument must be the accelerate config to use"
    exit 1
fi

MODEL=$2
OUTPUT=$3
DATA=$4
BATCH=$5
WORKERS=$6

# Disable internal parallelism of huggingface's tokenizer since we
# want to retain direct control of parallelism options.
export TOKENIZERS_PARALLELISM=false

NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)

export SINGULARITYENV_PREPEND_PATH=/user-software/bin

RUN_CMD="accelerate launch \
                    --config_file=$ACCELERATE_CONFIG \
                    --num_processes=$NUM_PROCESSES \
                    --num_machines=$SLURM_NNODES \
                    --machine_rank=\$SLURM_NODEID \
                    --main_process_ip=$MAIN_PROCESS_IP \
                    src/d_finetune/train_accelerate.py $* \
                    --model $MODEL \
                    --json-datapath /scratch/${SLURM_JOB_ACCOUNT}/${USER}/"$DATA" \
                    --output-path /scratch/${SLURM_JOB_ACCOUNT}/${USER}/"$OUTPUT" \
                    --num-workers $WORKERS \
                    --batch_size $BATCH \
                    --peft \
                    --4bit \
"

set -xv  # print the command so that we can verify setting arguments correctly from the logs

# Even though virtual environment is activated, the aforementioned error about EmbeddingParallel still occurs
srun singularity run $SIF bash -c "source ./ft_venv/bin/activate && $RUN_CMD"