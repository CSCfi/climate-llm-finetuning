#!/bin/bash
#SBATCH --account=project_xxx
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --output=./log/create_q/%j/output.log
#SBATCH --error=./log/create_q/%j/error.log
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=2:00:00
#SBATCH --nodes=1

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

export HF_HUB_CACHE=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub/
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
export TORCH_COMPILE_DISABLE=1

VLLM_LOG=$PWD/log/create_q/${SLURM_JOB_ID}/vllm.log
mkdir -p $(dirname $VLLM_LOG)

MODEL=$1
JSON_FILE_PATH=/scratch/${SLURM_JOB_ACCOUNT}/${SLURM_JOB_USER}/$2
SUFFIX=$3

singularity exec $SIF vllm serve $MODEL \
--tensor-parallel-size 8 \
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
    python src/c_qa/q_creation.py \
        --backend vllm \
        --model $MODEL \
        --filepath $JSON_FILE_PATH \
        --suffix $SUFFIX \
        --api-url http://0.0.0.0:8000/v1
"
Q_EXIT_CODE=$?

# Return the same exit code as q_creation.py
exit $Q_EXIT_CODE

kill $VLLM_PID
