#!/bin/bash 

if [ $# -lt 4 ]
then
	echo "usage:eval.sh <benchmark> <tracelen> <checkpoint> <policy> <opt:traindir(0|1)> <opt:testcsv>"
	exit 1
fi

bm=$1
len=$2
chkpt=$3

if [ $4 -eq 1 ]
then
	pol="lru"
elif [ $4 -eq 2 ]
then
	pol="belady"
else
	pol="learned"
fi

traindir="train/${bm}${len}"
if [ $# -gt 4 ]
then
	if [ $5 -eq 1 ]
	then
		traindir="${bm}_p100"
	fi
fi

if [ $# -eq 6 ]
then
	test=$6
else
	test="${bm}_${len}B_test"
fi

echo "#!/bin/bash"

echo "#BSUB -n 1"
echo "#BSUB -W 71:59"
echo "#BSUB -o logs/out.${bm}.eval.%J"
echo "#BSUB -e logs/err.${bm}.eval.%J"
echo "#BSUB -J hw1_${bm}_${pol}"

echo "source ~/.bashrc"
echo "conda activate env_pip_ver/"
if [ $4 -eq 0 ]; then
	echo "python3 -m cache_replacement.policy_learning.cache.main \\
	--experiment_base_dir=\"tmp/eval\" \\
	--experiment_name=\"${bm}${len}_chkpt${chkpt}k\" \\
	--cache_configs=\"cache_replacement/policy_learning/cache/configs/default.json\" \\
	--cache_configs=\"cache_replacement/policy_learning/cache/configs/eviction_policy/${pol}.json\" \\
	--memtrace_file=\"cache_replacement/policy_learning/cache/traces/${test}.csv\" \\
	--config_bindings=\"eviction_policy.scorer.checkpoint=\\\"tmp/${traindir}/checkpoints/${chkpt}000.ckpt\\\"\" \\
	--config_bindings=\"eviction_policy.scorer.config_path=\\\"tmp/${traindir}/model_config.json\\\"\" \\
	--warmup_period=0"
else
	echo "python3 -m cache_replacement.policy_learning.cache.main \\
	--experiment_base_dir=\"tmp/eval\" \\
	--experiment_name=\"${bm}${len}_${pol}\" \\
	--cache_configs=\"cache_replacement/policy_learning/cache/configs/default.json\" \\
	--cache_configs=\"cache_replacement/policy_learning/cache/configs/eviction_policy/${pol}.json\" \\
	--memtrace_file=\"cache_replacement/policy_learning/cache/traces/${test}.csv\""
fi
