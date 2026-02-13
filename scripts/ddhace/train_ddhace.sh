#!/usr/bin/env bash

model_name=DDHACE

python -u run.py \
  --is_training 1 \
  --model_id ddhace_demo \
  --model $model_name \
  --data custom \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 512 \
  --factor 1 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --train_epochs 10 \
  --exp_name ddhace_train
