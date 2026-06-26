#!/bin/bash
#SBATCH --account=project_xxx
#SBATCH --partition=dev-g
#SBATCH --output=./log/test_inference/%j/output.log
#SBATCH --error=./log/test_inference/%j/error.log
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

export HF_HUB_CACHE=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub/
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
export TORCH_COMPILE_DISABLE=1

VLLM_LOG=$PWD/log/test_inference/${SLURM_JOB_ID}/vllm.log
mkdir -p $(dirname $VLLM_LOG)

MODEL_NAME=$1
TEST_DATASET_PATH=/scratch/${SLURM_JOB_ACCOUNT}/${SLURM_JOB_USER}/$2
MAX_TOKENS=$3
BATCH_SIZE=$4
RESUME_CREATION=$5
SUFFIX=$6

if [[ "$MODEL_NAME" == climate-llm-finetuning/* ]]; then
    # Local finetuned model path
    MODEL_PATH="/scratch/${SLURM_JOB_ACCOUNT}/${SLURM_JOB_USER}/${MODEL_NAME}"
else
    # Hugging Face model name
    MODEL_PATH="$MODEL_NAME"
fi

singularity exec $SIF vllm serve $MODEL_PATH \
--tensor-parallel-size 1 \
--port 8000 > $VLLM_LOG &

VLLM_PID=$!

cleanup() {
    echo "Cleaning up vLLM process $VLLM_PID"
    kill $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting vLLM process $VLLM_PID - logs go to $VLLM_LOG"

# Wait until vLLM is running
sleep 20
while ! curl http://0.0.0.0:8000 >/dev/null 2>&1
do
    if [ -z "$(ps --pid $VLLM_PID --no-headers)" ]; then
        echo "vLLM crashed"
        exit 1
    fi
    sleep 10
done

# Run the actual Python job
singularity exec $SIF bash -c "
    export CUDA_VISIBLE_DEVICES='' && \
    python src/e_eval/model_test_inference.py \
        --model $MODEL_PATH \
        --filepath $TEST_DATASET_PATH \
        --model_suffix $SUFFIX \
        --max_tokens $MAX_TOKENS \
        --batch_size $BATCH_SIZE \
        --resume $RESUME_CREATION \
        --api-url http://0.0.0.0:8000/v1
"
Q_EXIT_CODE=$?

# Return the same exit code as a_creation.py
exit $Q_EXIT_CODE