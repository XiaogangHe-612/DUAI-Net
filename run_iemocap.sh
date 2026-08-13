#!/usr/bin/env bash
set -e

python train.py \
  --dataset IEMOCAP \
  --data-path ./data/iemocap_multimodal_features.pkl \
  --save-dir ./IEMOCAP \
  --seed 2094 \
  --lr 5e-5 \
  --dropout 0.425 \
  --l2 5e-5 \
  --batch-size 16 \
  --hidden-dim 512 \
  --n-head 8 \
  --epochs 100 \
  --windows 20 \
  --drp-warmup-epochs 0 \
  --stage1-aux-weight 0.10 \
  --edge-scale-gamma 0.45 \
  --edge-scale-alpha 4.0 \
  --edge-scale-beta 0.70 \
  --edge-logit-scale 0.45 \
  --dc-rank 2 \
  --darf-warmup-epochs 0 \
  --fusion-rank 1 \
  --audio-fusion-coeff 0.09 \
  --visual-fusion-coeff 0.01 \
  --darf-mode a
