import os
import tqdm
import torch
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DATASET = "imdb"
MODEL_NAME = "asun17904/imdb-bert-base-uncased"
BASE_MODEL_NAME = "bert-base-uncased"

dataset = load_dataset(DATASET)

test_dataset = dataset["test"]
shuffled_test = test_dataset.shuffle(seed=42)
test_dataset = shuffled_test.select(range(10000, len(test_dataset)))

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


# prepare the dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")


tokenized_test = test_dataset.map(tokenize, batched=True)

# run prediction on the test dataset, get raw prediction, label, and hidden states
outputs = []
model.eval()
model = model.to("cuda")
with torch.no_grad():
    for i in tqdm.tqdm(range(0, len(tokenized_test), 2)):
        batch = tokenized_test[i : i + 2]
        input_ids = torch.LongTensor(batch["input_ids"]).to("cuda")
        labels = torch.LongTensor(batch["label"]).to("cuda")
        attention_mask = torch.LongTensor(batch["attention_mask"]).to("cuda")
        outputs.append(
            model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )
        )

# save the output file
pickle.dump(
    outputs,
    open(
        f"/home/weicheng/data_interns/alan/kd/{BASE_MODEL_NAME}-{DATASET}-k0.pkl", "wb+"
    ),
)
