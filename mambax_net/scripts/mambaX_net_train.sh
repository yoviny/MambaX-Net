#!/usr/bin/env bash
# Wait until the GPU is sufficiently free before launching training.
# Polls every 60 s; starts when GPU memory usage drops below MEM_THRESHOLD_MB.
GPU_ID=${GPU_ID:-0}
MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB:-1000}
POLL_INTERVAL=${POLL_INTERVAL:-600}

wait_for_gpu() {
    echo "Waiting for GPU $GPU_ID to be free (threshold: < ${MEM_THRESHOLD_MB} MB used)..."
    while true; do
        USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null)
        if [ -z "$USED_MEM" ]; then
            echo "Error: nvidia-smi failed. Is a GPU available?"
            exit 1
        fi
        if [ "$USED_MEM" -lt "$MEM_THRESHOLD_MB" ]; then
            echo "GPU $GPU_ID is free (${USED_MEM} MB used). Starting training."
            break
        fi
        echo "  GPU $GPU_ID in use (${USED_MEM} MB used). Retrying in ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
    done
}

wait_for_gpu

# Parse --hpc flag to switch between local and HPC paths
HPC=false
for arg in "$@"; do
    if [ "$arg" = "--hpc" ]; then
        HPC=true
    fi
done

if [ "$HPC" = true ]; then
    XNAT_DIR="/scratch/users/k24001441/xnat_processed_2025"
    PROSTATE_DATA="/scratch/users/k24001441/Prostate_Data"
else
    XNAT_DIR="../ProstateCancer-AS/data_2025_v3/xnat_processed_2025"
    PROSTATE_DATA="../Prostate_Data"
fi

COMMON_ARGS="\
 -t2 $XNAT_DIR/t2 \
 -wp $PROSTATE_DATA/train_AI_course/wp \
 -pz $PROSTATE_DATA/train_AI_course/pz_tz \
 -val_t2 $XNAT_DIR/t2 \
 -val_wp $PROSTATE_DATA/valid/wp \
 -val_pz $PROSTATE_DATA/valid/pz_tz \
 -test_t2 $XNAT_DIR/t2 \
 -test_wp $PROSTATE_DATA/test/wp \
 -test_pz $PROSTATE_DATA/test/pz_tz \
 -best_preds $PROSTATE_DATA/train_AI_course/wp \
 -conf mambax_net/configs/train_xnat_mx.json \
 -nw 4"

# Fine-tune on AS data for increasing training set sizes
# sample_sz >= 250 uses all available patients (no sampling)
for SAMPLE_SZ in 5 10 50 100 250; do
    if [ "$SAMPLE_SZ" -eq 250 ]; then
        EXP_NAME="mxnet-AS-all"
    else
        EXP_NAME="mxnet-AS-${SAMPLE_SZ}"
    fi
    echo "=== Training $EXP_NAME (sample_sz=$SAMPLE_SZ) ==="
    ${PYTHON:-python} -m mambax_net.training.mx_net_train \
     $COMMON_ARGS \
     --sample_sz "$SAMPLE_SZ" --exp_name "$EXP_NAME"
done