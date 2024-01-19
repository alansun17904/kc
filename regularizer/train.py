import os
import torch
import torch.nn.functional as F
import numpy as np
import evaluate
import argparse
from datasets import load_dataset
from model import KnowledgeContinuousModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def prepare_dataset(tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
    tokenized_dataset = dataset.map(tokenize, batched=True)
    return tokenized_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def split_dataset(dataset):
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    shuffled_test = test_dataset.shuffle(seed=42)
    valid_dataset, test_dataset = shuffled_test.select(range(10000)), shuffled_test.select(range(10000, len(test_dataset)))
    return train_dataset, valid_dataset, test_dataset


# TODO: here we are not specifying a delta value, but we need to do this cause in the proofs we
# are doing this.
# TODO: also need to specify a lambda value on the regularizer term
class KnowledgeRegularizedTrainer(Trainer):
    def calc_knowledge_discontinuities(self, class_losses, hs):
        dist = torch.cdist(hs,hs) + 1
        loss_dist = torch.cdist(class_losses, class_losses, p=1)
        return torch.sum(loss_dist / dist)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits, hs = outputs
        if return_outputs:
            return logits
        class_loss = F.binary_cross_entropy(logits, labels, reduction="none")  # N x 1
        kd_score = self.calc_knowledge_discontinuities(class_loss, hs)
        return torch.sum(class_loss) + 0.02 * kd_score

def prepare_trainer(model, train_dataset, valid_dataset, epochs=20):
    training_args = TrainingArguments(
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        hub_token=os.environ.get("HUB_TOKEN")
        hub_model_id=f"imdb-kd-regularized"
        push_to_hub=True,
        seed=42
    )
    trainer = KnowledgeRegularizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )
    return trainer


options = argparse.ArgumentParser()
options.add_argument("dataset", type=str, help="any dataset that is in the huggingface dataset module")
options.add_argument("model", type=str, help="name of the model (huggingface repo)")
options.add_argument("alpha", type=float, help="parameter in the beta distribution for choosing hidden layer")
options.add_argument("beta", type=float, help="parameter in the beta distribution for choosing the hidden layer")
options.add_argument("-epochs", type=int, help="the number of training epochs")

options.parse_args()

metric = evaluate("accuracy")
tokenizer = AutoTokenizer.from_pretrained(options.model)
model = AutoModelForSequenceClassification.from_pretrained(options.model)
dataset = load_dataset(options.dataset)
train_dataset, valid_dataset, test_dataset = split_dataset(dataset)
train_dataset = prepare_dataset(train_dataset)
valid_dataset = prepare_dataset(valid_dataset)
trainer = prepare_dataset(
    KnowledgeContinuousModel(model, options.alpha, options.beta),
    train_dataset,
    valid_dataset
)
trainer.train()
regularized_model = trainer.model.model
# push this to hub too
regularized_model.push_to_hub(f"imdb-kd-regularized-base")



