"""
Given some summarization dataset, we preprocess the data based on its unique column names and
return train, validation, and test datasets. 
"""


from datasets import load_dataset



def preprocess_dataset(dataset_name, tokenizer):    
    # TODO: for t5, we need to add the prefix "summarize: " to the target text
    def preprocess_batch(examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        input_lens = [len(v) for v in inputs]
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length")

        labels = tokenizer(text_target=examples["highlights"], truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        
        model_inputs["s_idx"] = [v + 2 for v in input_lens]
        return model_inputs


    if dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        dataset = dataset.map(preprocess_batch, batched=True)
        # dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return dataset["train"], dataset["validation"], dataset["test"]

