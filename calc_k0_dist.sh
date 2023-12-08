#!/bin/bash

DATA_DIR="data/k0"
MODELS=(
	"bert-large-uncased-imdb"
	"roberta-base-imdb"
	"roberta-large-imdb"
)
LAYERS=(24 12 24)

for j in "${!MODELS[@]}";
do
	for i in `seq 0 ${LAYERS[$j]}`
	do
		python3.8 gpu_k0_dist.py data/k0 "${MODELS[$j]}" "$i"
	done
done
