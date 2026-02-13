#!/usr/bin/env bash

# Physics-Aware Control-SSM example training script (replace root_path/data_path with water-level data).
model_name=PhysicsAwareControlSSM

python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_PhysicsAware \
  --model ${model_name} \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --e_layers 2 \
  --n_heads 8 \
  --batch_size 32 \
  --learning_rate 0.0002 \
  --train_epochs 10 \
  --lambda_phys 0.05 \
  --lambda_lyap 0.1 \
  --lambda_rho 0.1 \
  --lyap_epsilon 1e-3 \
  --energy_alpha 0.1 \
  --rain_idx 0 \
  --flow_idx 1 \
  --wind_idx 2
