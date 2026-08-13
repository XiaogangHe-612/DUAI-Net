#!/usr/bin/env bash
set -e

python train.py \
  --dataset MELD \
  --data-path ./data/meld_multimodal_features.pkl \
  --save-dir ./MELD \
  --seed 2094 \
  --lr 3e-5 \
  --dropout 0.45 \
  --l2 5e-5 \
  --batch-size 16 \
  --hidden-dim 512 \
  --n-head 16 \
  --epochs 40 \
  --windows 5 \
  --no-class-weight \
  --drp-warmup-epochs 0 \
  --stage1-aux-weight 0.10 \
  --edge-scale-gamma 0.45 \
  --edge-scale-alpha 4.0 \
  --edge-scale-beta 0.50 \
  --edge-logit-scale 0.45 \
  --dc-rank 2 \
  --darf-warmup-epochs 0 \
  --fusion-rank 1 \
  --audio-fusion-coeff 0.05 \
  --visual-fusion-coeff 0.01 \
  --darf-mode av
