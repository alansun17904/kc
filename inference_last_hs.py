import os
import tqdm
import torch
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DATASET = "imdb"
MODEL_NAME = "asun17904/imdb-roberta-large"
BASE_MODEL_NAME = "roberta-large"

dataset = load_dataset(DATASET)

test_dataset = dataset["train"]
labels = [v["label"] for v in test_dataset]
pickle.dump(labels, open("/home/weicheng/data_interns/alan/kd/train_labels.pkl", "wb"))
# shuffled_test = test_dataset.shuffle(seed=42)
# test_dataset = shuffled_test.select(range(10000, len(test_dataset)))

model_name = MODEL_NAME 
tokenizer_name = BASE_MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# prepare the dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
tokenized_test = test_dataset.map(tokenize, batched=True)

# run prediction on the test dataset, get raw prediction, label, and hidden states
outputs = []
model.eval()
model = model.to("cuda")
with torch.no_grad():
    for i in tqdm.tqdm(range(0, len(tokenized_test), 2), desc="Model inference"):
        batch = tokenized_test[i:i+2]
        input_ids = torch.LongTensor(batch["input_ids"]).to("cuda")
        labels = torch.LongTensor(batch["label"]).to("cuda")
        attention_mask = torch.LongTensor(batch["attention_mask"]).to("cuda")
        lhs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                output_hidden_states=True,
                # output_attentions=True
        ).hidden_states[-1]
        # append the average across the sequence dimension
        outputs.append(torch.mean(lhs, axis=1))
    # concat all of the last hidden states together 
    # outputs = torch.cat(outputs)
    # then average across the sequence dimension
    # outputs = torch.mean(outputs, axis=1)
    
# save the output file 
pickle.dump(
    outputs,
    open(
        f"/home/weicheng/data_interns/alan/kd/{BASE_MODEL_NAME}-{DATASET}-train_hs.pkl",
        "wb+"
    )
)
