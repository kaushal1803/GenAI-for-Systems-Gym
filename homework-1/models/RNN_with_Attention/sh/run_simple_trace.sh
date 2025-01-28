#!/bin/bash
#BSUB -n 1
#BSUB -W 30
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -J evaluate_gems_checkpoint
#BSUB -o /share/ece592f24/acem_hw2/Azam/Imitation-Learning-for-Cache-Replacement/out/out.%J
#BSUB -e /share/ece592f24/acem_hw2/Azam/Imitation-Learning-for-Cache-Replacement/out/err.%J

source ~/.bashrc
conda activate /share/ece592f24/acem_hw2/env_hw2
cd /share/ece592f24/acem_hw2/Azam/Imitation-Learning-for-Cache-Replacement/

python3 -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=experiments \
  --experiment_name=sample_trace \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"log_likelihood\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
