#!/usr/bin/env bash
set -euo pipefail

# Example multi-step training pipeline for iTransformer TSE.
# Update dataset/model hyper-parameters for your experiment.

PYTHON=python
DATA_ROOT=./data/electricity/
DATA_FILE=electricity.csv
MODEL=iTransformer
DATASET=custom
FEATURES=M
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96
ENC_IN=321
DEC_IN=321
C_OUT=321
BATCH=32
EPOCHS=10
LR=0.0001

COMMON_ARGS=(
  --is_training 1
  --model_id tse_exp
  --model "$MODEL"
  --data "$DATASET"
  --root_path "$DATA_ROOT"
  --data_path "$DATA_FILE"
  --features "$FEATURES"
  --seq_len "$SEQ_LEN"
  --label_len "$LABEL_LEN"
  --pred_len "$PRED_LEN"
  --enc_in "$ENC_IN"
  --dec_in "$DEC_IN"
  --c_out "$C_OUT"
  --batch_size "$BATCH"
  --train_epochs "$EPOCHS"
  --learning_rate "$LR"
)

# Step 1: Train baseline Adam model A
$PYTHON run.py "${COMMON_ARGS[@]}" \
  --model_id tse_parent_a \
  --optimizer adam \
  --des tse_stage0_parent_a

# Step 2: Train SAM/ASAM model B (choose one)
$PYTHON run.py "${COMMON_ARGS[@]}" \
  --model_id tse_parent_b \
  --optimizer asam \
  --sam_rho 0.05 \
  --des tse_stage0_parent_b

# Step 3: Run TSE twin-regularized optimization from A/B checkpoints
# Replace CKPT_A and CKPT_B with actual best checkpoint paths generated in ./checkpoints/
CKPT_A=./checkpoints/tse_parent_a_${MODEL}_${DATASET}_${FEATURES}_ft${SEQ_LEN}_sl${LABEL_LEN}_ll${PRED_LEN}_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_tse_stage0_parent_a_projection_0/checkpoint.pth
CKPT_B=./checkpoints/tse_parent_b_${MODEL}_${DATASET}_${FEATURES}_ft${SEQ_LEN}_sl${LABEL_LEN}_ll${PRED_LEN}_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_tse_stage0_parent_b_projection_0/checkpoint.pth

$PYTHON run.py "${COMMON_ARGS[@]}" \
  --model_id tse_stage1 \
  --optimizer tse \
  --sam_rho 0.05 \
  --tse_parent_a "$CKPT_A" \
  --tse_parent_b "$CKPT_B" \
  --tse_importance 10.0 \
  --tse_alpha 0.9 \
  --tse_fisher_batches 20 \
  --des tse_stage1_alpha09

# Step 4: optional second TSE stage by swapping to stage1 outputs as new parents.
