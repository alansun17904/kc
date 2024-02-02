import tqdm
import json
import torch
import argparse
import evaluate
from torch.utils.data import DataLoader
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    options.model, num_label=num_labels
)

# first we run inference on the entire test set without any perturbations and check the accuracy
dataset = load_dataset("glue", options.task)
# get the test dataset
test_dataset = dataset["test"]
test_dataset = prepare_dataset(test_dataset, tokenizer, options.task, is_gpt=options.model=="gpt2")
# run inference on test set and report the raw accuracy
dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for batch in tqdm.tqdm(test_dataset):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        output = pretrained_model(input_ids)
        logits = output.logits
        predictions = torch.argmax(logits, dim=1)
        all_labels.extend(labels)
        all_predictions.extend(predictions)
print("accuracy no attack", metric.compute(predictions=all_predictions, references=all_labels))

# now run infernece on adversarial glue
datapoints = json.load(open("test_ann.json"))[options.task]
# create a new dataset object from this 
adv_dataset = Dataset.from_list(datapoints)
adv_dataset = prepare_dataset(adv_dataset, tokenizer, options.task, is_gpt=options.model=="gpt2")
dataloader = DataLoader(adv_dataset, batch_size=4, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.eval()
all_adv_predictions = []
all_adv_labels = []
with torch.no_grad():
    for batch in tqdm.tqdm(test_dataset):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        output = pretrained_model(input_ids)
        logits = output.logits
        predictions = torch.argmax(logits, dim=1)
        all_adv_labels.extend(labels)
        all_adv_predictions.extend(predictions)
print("accuracy under attack", metric.compute(predictions=all_adv_predictions, references=all_adv_labels))