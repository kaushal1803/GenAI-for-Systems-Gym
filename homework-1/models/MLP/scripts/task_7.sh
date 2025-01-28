#!/bin/bash 



rep=("lru" "ship" "srrip" "drrip")
trace=("lbm_564B" "libquantum_1210B" "mcf_158B" )

for r in ${rep[@]}; do
  for t in ${trace[@]}; do
  bsub -n 1 -W 12:00 -o logs/${r}.${t}.out.%J -e logs/${r}.${t}.err.%J -J champsim_${r}_${t} bin/champsim_${r} --warmup_instructions 200000000 --simulation_instructions 1000000000 /share/ece592f24/group-a/ngopala2/trace/${t}.trace.xz
done
done