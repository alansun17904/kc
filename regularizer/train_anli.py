import os
import torch
import evaluate
import torch.nn.functional as F
import numpy as np
import argparse
from datasets import load_dataset
from model import KnowledgeContinuousModel
from huggingface_hub import ModelCard, create_repo
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)


metric = evaluate.load("accuracy")


def prepare_dataset(dataset, tokenizer, is_gpt=False):
    if is_gpt:
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], padding="max_length", truncation=True)
    tokenized_dataset = dataset.map(tokenize, batched=True)
    return tokenized_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def split_dataset(dataset, round):
    return (
        dataset[f"train_r{round}"],
        dataset[f"dev_r{round}"],
        dataset[f"test_r{round}"]
    )


class LoggingCallback(TrainerCallback):
    def __init__(self, alpha, beta, lam, normalizer):
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.normalizer = normalizer

    def on_epoch_end(self, args, state, control, **kwargs):
        content = f"""
---
language: en
license: mit
library_name: pytorch
---
# Knowledge Continuity Regularized Network
Dataset: ANLI
Round: {kwargs.get('round', None)}

Trainer Hyperparameters:
- `lr` = {args.learning_rate}
- `per_device_batch_size` = {args.per_device_train_batch_size}
- `gradient_accumulation_steps` = {args.gradient_accumulation_steps}
- `weight_decay` = {args.weight_decay}
- `seed` = {args.seed}

Regularization Hyperparameters
- `numerical stability denominator constant` = {self.normalizer}
- `lambda` = {self.lam}
- `alpha` = {self.alpha}
- `beta` = {self.beta}

Extended Logs:

|eval_loss|eval_accuracy|epoch|
|--|--|--|
"""
        for epoch in state.log_history:
            if "eval_loss" in epoch:
                content += f"|{epoch['eval_loss']:.3f}|{epoch['eval_accuracy']:.3f}|{epoch['epoch']}|\n"
        card = ModelCard(content)
        card.push_to_hub(f"asun17904/{args.hub_model_id}", token=os.environ["HUB_TOKEN"])


class KnowledgeRegularizedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            labels = inputs.get("labels", None)
            prediction_loss, model_output = self.compute_loss(
                model, inputs, return_outputs=True
            )
            _, logits = model_output
            if prediction_loss_only:
                return (prediction_loss, None, None)
            return (prediction_loss, logits, labels)

    def calc_knowledge_discontinuities(self, class_losses, hs):
        dist = torch.cdist(hs, hs) + self.stabilizer
        loss_dist = torch.cdist(class_losses, class_losses, p=1)
        return torch.sum(loss_dist / dist)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        hs, logits = outputs
        logits = logits.softmax(dim=1)
        class_loss = F.cross_entropy(logits, labels, reduction="none")  # N x 1
        class_loss = class_loss.reshape(-1, 1)
        kd_score = self.calc_knowledge_discontinuities(class_loss, hs)
        if return_outputs:
            return torch.sum(class_loss) + self.lam * kd_score, outputs
        return torch.sum(class_loss) + self.lam * kd_score


def prepare_trainer(
    round_number,
    model_name,
    model,
    train_dataset,
    valid_dataset,
    alpha,
    beta,
    lam,
    stabilizer,
    learning_rate,
    weight_decay,
    epochs=20,
):
    training_args = TrainingArguments(
        output_dir=f"anliR{round_number}-{model_name}-reg",
        per_device_train_batch_size=16,
        # gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        # eval_accumulation_steps=4,
        weight_decay=weight_decay,
        hub_token=os.environ.get("HUB_TOKEN"),
        hub_model_id=f"anliR{round_number}-{model_name}-reg",
        push_to_hub=True,
        save_steps=2000,
        seed=42,
    )
    trainer = KnowledgeRegularizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback(alpha, beta, lam, stabilizer)],
    )
    trainer.lam = lam
    trainer.stabilizer = stabilizer
    return trainer


parser = argparse.ArgumentParser()
parser.add_argument("round", type=int, help="round of anli task")
parser.add_argument("model", type=str, help="name of the model (huggingface repo)")
parser.add_argument(
    "alpha",
    type=float,
    help="parameter in the beta distribution for choosing hidden layer",
)
parser.add_argument(
    "beta",
    type=float,
    help="parameter in the beta distribution for choosing the hidden layer",
)
parser.add_argument("lam", type=float, help="weight given to the regularization term")
parser.add_argument("stabilizer", type=float, help="stabilizer term")
parser.add_argument("learning_rate", type=float, help="learning rate")
parser.add_argument("weight_decay", type=float, help="weight decay")
parser.add_argument("-epochs", type=int, help="the number of training epochs")
parser.add_argument("-is_ed", type=bool, help="if the model is an encoder-decoder", default=False)

options = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(options.model)
pretrained_model = AutoModelForSequenceClassification.from_pretrained(options.model, num_labels=3)

# set the padding token if the model is gpt
if options.model == "gpt2":
    pretrained_model.config.pad_token_id = pretrained_model.config.eos_token_id

dataset = load_dataset("anli")
train_dataset, valid_dataset, test_dataset = split_dataset(dataset, options.round)
train_dataset = prepare_dataset(train_dataset, tokenizer, is_gpt=options.model=="gpt2")
valid_dataset = prepare_dataset(valid_dataset, tokenizer, is_gpt=options.model=="gpt2")
test_dataset = prepare_dataset(valid_dataset, tokenizer, is_gpt=options.model=="gpt2")

# train_dataset = train_dataset.select(range(20))
# valid_dataset = valid_dataset.select(range(20))
# test_dataset = test_dataset.select(range(20))


trainer = prepare_trainer(
    options.round,
    options.model,
    KnowledgeContinuousModel(
        pretrained_model,
        options.alpha,
        options.beta,
        options.is_ed,
    ),
    train_dataset,
    valid_dataset,
    options.alpha,
    options.beta,
    options.lam,
    options.stabilizer,
    options.learning_rate,
    options.weight_decay,
    epochs=options.epochs,
)
# trainer.evaluate()
trainer.train()
test_results = trainer.predict(test_dataset)
print(test_results)
card = ModelCard.load(f"asun17904/anliR{options.round}-{options.model}-reg")
card.text += f"\n**Test Accuracy: {test_results.metrics['test_accuracy']:.3f}**"
card.push_to_hub(f"asun17904/anliR{options.round}-{options.model}-reg", token=os.environ["HUB_TOKEN"])

# regularized_model = trainer.model.model
# push this to hub too
# regularized_model.save_pretrained(f"imdb-{options.model}-kd-regularized-base-l2")
