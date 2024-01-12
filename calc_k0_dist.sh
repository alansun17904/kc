#!/bin/bash

DATA_DIR="data/k0"
MODELS=(
	"xlnet-base-cased"
)
LAYERS=(13)

for j in "${!MODELS[@]}";
do
	for i in `seq 0 ${LAYERS[$j]}`
	do
		python gpu_k0_dist.py data/k0 "${MODELS[$j]}" "$i"
	done
done
