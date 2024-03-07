import os
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from textattack import Attacker, AttackArgs
import textattack.attack_recipes as recipes
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper

from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)

import csv

from timeout_decorator import timeout

import traceback

parser = argparse.ArgumentParser()
parser.add_argument("split", type=str, help="train/test")
parser.add_argument("model_name", type=str, help="hugging face model repo id")
parser.add_argument("base_model_name", type=str, help="name of the base model")
parser.add_argument("check_point_dir", type=str, help="directory to store check points")
parser.add_argument("output_filename", type=str, help="name of the output file")
parser.add_argument(
    "-marking_style",
    type=str,
    help="marking style for adv perturbations",
    default="file",
)

options = parser.parse_args()

DATASET = "imdb"
MODEL_NAME = options.model_name
BASE_MODEL_NAME = options.base_model_name
CHECKPOINT_DIR = options.check_point_dir
RECIPE = recipes.bert_attack_li_2020.BERTAttackLi2020

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
origin_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

if BASE_MODEL_NAME == "gpt2":
    tokenizer.pad_token = tokenizer.eos_token

model = HuggingFaceModelWrapper(origin_model, tokenizer)
dataset = HuggingFaceDataset(DATASET, None, options.split, shuffle=True)
model.to('cuda')

attack_args = AttackArgs(
    num_examples=5000,
    checkpoint_interval=5,
    checkpoint_dir=CHECKPOINT_DIR,
    csv_coloring_style=options.marking_style,
    log_to_csv=f"{options.output_filename}.csv",
    query_budget=300,
    parallel=True,
    # num_workers_per_device=2,
    # timeout=60
)

import time

@timeout(120)
def attack_with_timeout(text, attacker, ground_truth):  # Adjust timeout as needed
    print(text)
    attack_result = attacker.attack(text, ground_truth)
    return attack_result

def attack_loop(dataset, attacker, output_file):
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "original_text",
            "perturbed_text",
            "original_score",
            "perturbed_score",
            "original_output",
            "perturbed_output",
            "ground_truth_output",
            "num_queries",
            "result_type"
        ])
        for text_input, label in dataset:  # Access text and optional labels
            try:
                print(label)
                attack_result = attack_with_timeout(text_input['text'], attacker, label)
                # Process the attack results as needed (e.g., print, store in a list)
                print(f"Attack result for '{text_input}': {attack_result}")

                original_text = text_input['text']
                perturbed_text = attack_result.perturbed_text()
                original_score = attack_result.original_result.score  # You might need to modify this based on your dataset format
                perturbed_score = attack_result.perturbed_result.score  # You might need to modify this based on your dataset format
                original_output = attack_result.original_result.output  # You might need to modify this based on your dataset format
                perturbed_output = attack_result.perturbed_result.output  # Assuming 'result' field contains the output
                ground_truth_output = label  # Assuming label is stored in "label" key
                num_queries = attack_result.num_queries
                result_type = "" #attack_result.attack_result_type

                if isinstance(attack_result, FailedAttackResult):
                    result_type = "Failed"
                elif isinstance(attack_result, SkippedAttackResult):
                    result_type = "Skipped"
                elif isinstance(attack_result, SuccessfulAttackResult):
                    result_type = "Success"
                else:
                    result_type = "Unknown"

                # Write data to CSV
                writer.writerow([
                    original_text,
                    perturbed_text,
                    original_score,
                    perturbed_score,
                    original_output,
                    perturbed_output,
                    ground_truth_output,
                    num_queries,
                    result_type
                ])

                csvfile.flush()
            except Exception as e:
                print(f"Error encountered while attacking '{text_input}': {e}")
                print(traceback.format_exc())

                writer.writerow([
                    text_input['text'],
                    "",
                    "",
                    "",
                    "",
                    "",
                    label,
                    300,
                    "TLE"])

                csvfile.flush()


    # Additional steps after all examples are processed (e.g., save results)


if __name__ == "__main__":
    #attacker = Attacker(RECIPE.build(model), dataset, attack_args)
    try:
        attacker = RECIPE.build(model)
        attacker.cuda_()
        attacker.goal_function.query_budget = 300
        attack_loop(dataset, attacker, f"{options.output_filename}.csv")
    except:
        print(traceback.format_exc())