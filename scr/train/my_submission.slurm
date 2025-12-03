#!/bin/bash
#SBATCH -J predict_epitope       
#SBATCH -A PAS2177         
#SBATCH --gpus=2   
#SBATCH --mem=128G          
#SBATCH -t 3:00:00       
#SBATCH -o predict_epitope.%j.out  
#SBATCH -e predict_epitope.%j.err  
python /users/PAS2177/liu9756/DualChain-SelfAg_prediction/scr/train/train_epitope_classifier.py --data /users/PAS2177/liu9756/DualChain-SelfAg_prediction/data/featurized_min5 --outdir /users/PAS2177/liu9756/DualChain-SelfAg_prediction/results/train_results_min5 --epochs 1000 --bs 128 --lr 1e-4 --wd 2e-4 --d_model 256 --use_hla 1 --use_seq_a 1 --use_seq_b 1 --use_cosine_head 1 --use_focal 1 --label_smoothing 0 --use_sampler 1 --sched cosine --early_stop 1 --early_stop_patience 50 --temp_scale 1 