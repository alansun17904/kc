import numpy as np
import tqdm
import json
import torch
import argparse
import evaluate
from torch.utils.data import DataLoader
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

metric = evaluate.load("accuracy")

def prepare_dataset(dataset, tokenizer, task, is_gpt=False):
    if is_gpt:
        tokenizer.pad_token = tokenizer.eos_token
    if task == "mnli":

        def tokenize(batch):
            return tokenizer(
                batch["premise"],
                batch["hypothesis"],
                padding="max_length",
                truncation=True,
            )

    elif task == "cola":

        def tokenize(batch):
            return tokenizer(batch["sentence"], padding="max_length", truncation=True)

    elif task == "mrpc" or task == "wnli" or task == "rte":

        def tokenize(batch):
            return tokenizer(
                batch["sentence1"],
                batch["sentence2"],
                padding="max_length",
                truncation=True,
            )

    elif task == "qqp":

        def tokenize(batch):
            return tokenizer(
                batch["question1"],
                batch["question2"],
                padding="max_length",
                truncation=True,
            )

    elif task == "qnli":

        def tokenize(batch):
            return tokenizer(
                batch["question"],
                batch["sentence"],
                padding="max_length",
                truncation=True,
            )

    tokenized_dataset = dataset.map(tokenize, batched=True)
    return tokenized_dataset


parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="model id on huggingface")
parser.add_argument("base_model_name", type=str, help="name of the base model huggingface model id")
parser.add_argument("task", type=str, help="the name of the task being benchmarked against")
options = parser.parse_args()


if options.task == "qqp":
    num_labels = 2
else:
    num_labels = 3

tokenizer = AutoTokenizer.from_pretrained(options.base_model_name)
pretrained_model = AutoModelForSequenceClassification.from_pretrained(
    options.model_name, num_labels=num_labels
)

# first we run inference on the entire test set without any perturbations and check the accuracy
dataset = load_dataset("glue", options.task)
# get the test dataset
# now run infernece on adversarial glue
datapoints = json.load(open("regularizer/test_ann.json"))[options.task]
# create a new dataset object from this 
adv_dataset = Dataset.from_list(datapoints)
adv_dataset = prepare_dataset(adv_dataset, tokenizer, options.task, is_gpt=options.base_model_name=="gpt2")
train_args = TrainingArguments(
    "inference",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
)
trainer = Trainer(
    model=pretrained_model,
    args=train_args,
    train_dataset=adv_dataset,
    eval_dataset=adv_dataset,
    tokenizer=tokenizer
)

predictions = trainer.predict(adv_dataset)
predictions = np.argmax(predictions.predictions,axis=1)
print(
    "accuracy under attack",
    metric.compute(
        predictions=predictions, 
        references=[v["label"] for v in adv_dataset]
    )
)

