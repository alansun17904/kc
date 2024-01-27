import os
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from textattack import Attacker, AttackArgs
import textattack.attack_recipes as recipes
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="hugging face dataset id")
parser.add_argument("task", type=str, help="glue task to target")
parser.add_argument("model_name", type=str, help="hugging face model repo id")
parser.add_argument("base_model_name", type=str, help="name of the base model")
parser.add_argument("check_point_dir", type=str, help="directory to store check points")

options = parser.parse_args()

DATASET = options.dataset
TASK = options.task
MODEL_NAME = options.model_name
BASE_MODEL_NAME = options.base_model_name
CHECKPOINT_DIR = options.check_point_dir
RECIPE = recipes.PWWSRen2019

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

if BASE_MODEL_NAME == "gpt2":
    tokenizer.pad_token = tokenizer.eos_token

origin_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model = HuggingFaceModelWrapper(origin_model, tokenizer)
dataset = HuggingFaceDataset(DATASET, TASK, "test", shuffle=False)

attack_args = AttackArgs(
    num_examples=-1,
    checkpoint_interval=1000,
    checkpoint_dir=CHECKPOINT_DIR,
    log_to_csv=f"ba-{BASE_MODEL_NAME}-{TASK}.csv",
    query_budget=300,
    # parallel=True,
    # num_workers_per_device=4
)

attacker = Attacker(RECIPE.build(model), dataset, attack_args)

# print(attacker.goal_function)
# attacker.attack.constraints[2].skip_text_shorter_than_window = False

attacker.attack_dataset()
