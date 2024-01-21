import os
import torch
import evaluate
import torch.nn.functional as F
import numpy as np
import argparse
from datasets import load_dataset
from model import KnowledgeContinuousModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


metric = evaluate.load("accuracy")

def prepare_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            labels = inputs.get("labels", None)
            prediction_loss, model_output = self.compute_loss(model, inputs, return_outputs=True)
            _, logits = model_output
            if prediction_loss_only:
                return (prediction_loss, None, None)
            return (prediction_loss, logits, labels)

    def calc_knowledge_discontinuities(self, class_losses, hs):
        dist = torch.cdist(hs,hs) + 1e-2
        loss_dist = torch.cdist(class_losses, class_losses, p=1)
        return torch.sum(loss_dist / dist)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        hs, logits = outputs
        labels = F.one_hot(labels,num_classes=2).float()
        logits = logits.softmax(dim=1)
        class_loss = F.cross_entropy(logits, labels, reduction="none")  # N x 1
        class_loss = class_loss.reshape(-1, 1)
        kd_score = self.calc_knowledge_discontinuities(class_loss, hs)
        if return_outputs:
            return torch.sum(class_loss) + 5e-3 * kd_score, outputs
        return torch.sum(class_loss) + 5e-3 * kd_score

def prepare_trainer(model_name, model, train_dataset, valid_dataset, epochs=20):
    training_args = TrainingArguments(
        output_dir="imdb-kd-regularized",
        per_device_train_batch_size=8,
        #gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        # eval_accumulation_steps=4,
        hub_token=os.environ.get("HUB_TOKEN"),
        hub_model_id=f"imdb-{model_name}-kd-regularized",
        push_to_hub=True,
        save_steps=2000,
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


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="any dataset that is in the huggingface dataset module")
parser.add_argument("model", type=str, help="name of the model (huggingface repo)")
parser.add_argument("alpha", type=float, help="parameter in the beta distribution for choosing hidden layer")
parser.add_argument("beta", type=float, help="parameter in the beta distribution for choosing the hidden layer")
parser.add_argument("-epochs", type=int, help="the number of training epochs")

options = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(options.model)
pretrained_model = AutoModelForSequenceClassification.from_pretrained(options.model)
dataset = load_dataset(options.dataset)
train_dataset, valid_dataset, test_dataset = split_dataset(dataset)
train_dataset = prepare_dataset(train_dataset, tokenizer)
valid_dataset = prepare_dataset(valid_dataset, tokenizer)
trainer = prepare_trainer(
    options.model,
    KnowledgeContinuousModel(pretrained_model, options.alpha, options.beta),
    train_dataset,
    valid_dataset,
    epochs=options.epochs
)
# trainer.evaluate()
trainer.train()
regularized_model = trainer.model.model
# push this to hub too
regularized_model.save_pretrained(f"imdb-{options.model}-kd-regularized-base")



