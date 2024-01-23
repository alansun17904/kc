import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from textattack import Attacker, AttackArgs
import textattack.attack_recipes as recipes
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper


DATASET = "asun17904/imdb-test"
MODEL_NAME = "asun17904/imdb-bert-base-uncased-kd-regularized"
BASE_MODEL_NAME = "bert-base-uncased"
CHECKPOINT_DIR = "data/adversarial-attacks/"
RECIPE = recipes.TextFoolerJin2019

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
origin_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model = HuggingFaceModelWrapper(origin_model, tokenizer)
dataset = HuggingFaceDataset(DATASET, None, "test", shuffle=False)

attack_args = AttackArgs(
    num_examples=-1,
    checkpoint_interval=1000,
    checkpoint_dir=CHECKPOINT_DIR,
    log_to_csv=f"tf-{BASE_MODEL_NAME}-regularized.csv",
    query_budget=300,
    parallel=True,
    num_workers_per_device=4,
)

attacker = Attacker(RECIPE.build(model), dataset, attack_args)

attack.attack_dataset()
