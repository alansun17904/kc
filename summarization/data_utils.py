"""
Given some summarization dataset, we preprocess the data based on its unique column names and
return train, validation, and test datasets. 
"""


from datasets import load_datasets


def preprocess_dataset(dataset_name, tokenizer):    
    # TODO: for t5, we need to add the prefix "summarize: " to the target text
    if dataset_name == "cnn_dailymail":
        dataset = load_datasets("cnn_dailymail")
        dataset = dataset.map(
            lambda x: tokenizer(x["article"], x["highlights"], truncation=True, padding="max_length"),
            batched=True,
        )
        # dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return dataset["train"], dataset["validation"], dataset["test"]

