#!/bin/bash 

if [ $# -ne 2 ]
then
	echo "usage:train.sh <benchmark> <tracelen>"
	exit 1
fi

bm=$1
len=$2

echo "#!/bin/bash"

echo "#BSUB -n 4"
echo "#BSUB -W 71:59"
echo "#BSUB -q gpu"
echo "#BSUB -gpu \"num=1:mode=shared:mps=yes\""
echo "#BSUB -o logs/out.${bm}.train.%J"
echo "#BSUB -e logs/err.${bm}.train.%J"
echo "#BSUB -J hw1_${bm}${len}"
echo "#BSUB -R \"select[p100]\""

echo "source ~/.bashrc"
echo "conda activate env_pip_ver/"
echo "python3 -m cache_replacement.policy_learning.cache_model.main \\
	--experiment_base_dir=\"tmp/train\" \\
	--experiment_name=${bm}${len} \\
	--cache_configs=\"cache_replacement/policy_learning/cache/configs/default.json\" \\
	--model_bindings=\"loss=[\\\"ndcg\\\", \\\"reuse_dist\\\"]\" \\
  	--model_bindings=\"address_embedder.max_vocab_size=5000\" \\
	--train_memtrace=cache_replacement/policy_learning/cache/traces/${bm}_${len}B_train.csv \\
	--valid_memtrace=cache_replacement/policy_learning/cache/traces/${bm}_${len}B_valid.csv"
