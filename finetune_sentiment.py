import os
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments


dataset = load_dataset('imdb')
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def load_model_tokenizer(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def split_dataset(dataset):
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    shuffled_test = test_dataset.shuffle(seed=42)
    valid_dataset, test_dataset = shuffled_test.select(range(10000)), shuffled_test.select(range(10000, len(test_dataset)))
    return train_dataset, valid_dataset, test_dataset

def prepare_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_datasets = dataset.map(tokenize, batched=True)
    return tokenized_datasets

def prepare_trainer(model_name, model, train_dataset, valid_dataset):
    training_args = TrainingArguments(
        num_train_epochs=20,
        evaluation_strategy="epoch",
        hub_token=os.environ.get("HUB_TOKEN"),
        hub_model_id=f"imdb-{model_name}",
        output_dir=f"imdb-{model_name}",
        push_to_hub=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    return trainer

def main():
    models = ["xlnet-base-cased"]
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset)
    for model_name in models:
        model, tokenizer = load_model_tokenizer(model_name)
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        valid_dataset = prepare_dataset(valid_dataset, tokenizer)
        test_dataset = prepare_dataset(test_dataset, tokenizer)
        trainer = prepare_trainer(model_name, model, train_dataset, valid_dataset)
        trainer.train()


if __name__ == "__main__":
    main()
