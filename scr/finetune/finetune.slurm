#!/bin/bash
#SBATCH -J ft_epitope
#SBATCH -A PAS2177
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o ft_epitope.%j.out
#SBATCH -e ft_epitope.%j.err

python /users/PAS2177/liu9756/DualChain-SelfAg_prediction/scr/finetune/finetune_epitope_classifier.py \
  --data /users/PAS2177/liu9756/DualChain-SelfAg_prediction/data/featurized_ft \
  --pretrained /users/PAS2177/liu9756/DualChain-SelfAg_prediction/results/train_results_min5/best_acc.pt \
  --outdir /users/PAS2177/liu9756/DualChain-SelfAg_prediction/results/finetune_results_min5_with_record \
  --epochs 500 --bs 64 \
  --d_model 256 \
  --lr_backbone 1e-5 --lr_head 5e-4 \
  --use_sampler 0 \
  --use_focal 1 \
  --label_smoothing 0.05 \
  --early_stop 0 --early_stop_patience 100