#!/bin/bash
#BSUB -n 1
#BSUB -W 48:00
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -J evaluate_gems_checkpoint
#BSUB -o /share/ece592f24/acem_hw2/Azam/Imitation-Learning-for-Cache-Replacement/out/out.%J
#BSUB -e /share/ece592f24/acem_hw2/Azam/Imitation-Learning-for-Cache-Replacement/out/err.%J

source ~/.bashrc
conda activate /share/ece592f24/acem_hw2/env_hw2
cd /share/ece592f24/acem_hw2/Azam/Imitation-Learning-for-Cache-Replacement/

python3 -m cache_replacement.policy_learning.cache.main \
  --experiment_base_dir=experiments \
  --experiment_name=eval_lbm_564B_v2 \
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" \
  --memtrace_file="cache_replacement/policy_learning/cache/traces/lbm_564B_test.csv" \
  --config_bindings="associativity=16" \
  --config_bindings="capacity=2097152" \
  --config_bindings="eviction_policy.scorer.checkpoint=\"experiments/lbm_564B_v1/checkpoints/40000.ckpt\"" \
  --config_bindings="eviction_policy.scorer.config_path=\"experiments/lbm_564B_v1/model_config.json\"" \
  --warmup_period=0
