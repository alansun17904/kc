#!/bin/bash

DATA_DIR="data/k0"
MODELS=(
	"bert-large-uncased-imdb"
	"bert-base-uncased-imdb"
	"roberta-large-imdb"
	"roberta-base-imdb"
)
LAYERS=(25 13 25 13)

for i in `seq 0 ${#MODELS}`
do
	echo "Running with arguments" "${MODELS[i]}" ${LAYERS[i]}
	python3.6 visuals/layer_by_layer_dist.py data/k0/discontinuities/ ${MODELS[i]} ${LAYERS[i]}
done
