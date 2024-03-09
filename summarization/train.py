import os
import argparse

import torch
import evaluate
import numpy as np
import torch.nn.functional as F
from transformers import Trainer
from transformers import GPT2LMHeadModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.profiler import profile, record_function, ProfilerActivity
from peft import LoraConfig, TaskType
from peft import get_peft_model

from data_utils import preprocess_dataset

# Create a logger that logs all of the hyperparameters, the validation loss/accuracy
# at each epoch, and the final test loss/accuracy. It also takes in a dictionary 
# and logs these. 

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="name of the model, huggingface repository")
parser.add_argument("--kd", type=bool, help="if the model is regularized")
parser.add_argument("--alpha", type=float, help="alpha parameter in the beta distribution")
parser.add_argument("--beta", type=float, help="beta parameter in the beta distribution")
parser.add_argument("--lam", type=float, help="weight given to the regularizer")
parser.add_argument("--epochs", type=int, help="number of training epochs")
parser.add_argument("--learning_rate", type=float, help="learning rate", default=5e-5)
parser.add_argument("--weight_decay", type=float, help="weight decay", default=0)

options = parser.parse_args()

# create the evaluation function
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

# create the tokenizer and model 
tokenizer = AutoTokenizer.from_pretrained(options.model_name)

if "gpt2" in options.model_name:
    tokenizer.pad_token = tokenizer.eos_token


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # perform argmax to get the token ids
    # predictions = np.argmax(predictions, axis=2)

    # print(predictions)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print(decoded_preds)
    print(decoded_labels)

    result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    # result["gen_len"] = np.mean(prediction_lens)

    joint_results = list(result_bleu.items())
    joint_results.extend(result_rouge.items())

    return {k: v for (k, v) in joint_results}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[1], dim=-1)
    # pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


class SummarizationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        if kwargs.get("kd", False):
            self.regularization = True
            self.alpha = kwargs.get("alpha")
            self.beta = kwargs.get("beta")
            self.lam = kwargs.get("lam")
        else:
            self.regularization = False
        del kwargs["alpha"]
        del kwargs["beta"]
        del kwargs["lam"]
        del kwargs["kd"]
        super().__init__(*args, **kwargs)

    def calc_knowledge_discontinuities(self, logit_loss, hs):
        # p, q = np.random.randint(len(logit_loss)), np.random.randint(len(logit_loss))
        dist = torch.cdist(hs, hs) + 1e-2
        loss_dist = torch.cdist(logit_loss, logit_loss, p=1)
        return torch.sum(loss_dist / dist)
        # return torch.sum((logit_loss[p] - logit_loss[q]) ** 2 / torch.abs(hs[p] - hs[q]))

    def shift_output_logits(self, logits, inputs):
        s_idx = inputs.get("s_idx")  # separating index
        shifted_logits = logits[...,s_idx:,:].contiguous()
        return shifted_logits 

    def compute_loss(self, model, inputs, return_outputs=False):
        # check if the model is an encoder-decoder model
        ed = True # model.config.is_encoder_decoder
        if not self.regularization or return_outputs:
            # print(model(**inputs).logits.shape)
            return super().compute_loss(model, inputs, return_outputs)
        labels = inputs.get("labels")
        # get all of the hidden states and output from the model
        outputs = model(
            **inputs,
            output_hidden_states=True
        )
        if ed:
            num_encoder_hs, num_decoder_hs = (
                len(outputs.encoder_hidden_states),
                len(outputs.decoder_hidden_states),
            )
            layer = int(
                (num_encoder_hs + num_decoder_hs) * np.random.beta(self.alpha, self.beta)
            )
            if layer >= num_encoder_hs:
                hs = torch.mean(outputs.decoder_hidden_states[layer - num_encoder_hs],axis=1)
            else:
                hs = torch.mean(outputs.encoder_hidden_states[layer], axis=1)
            # del outputs.encoder_hidden_states
            # del outputs.decoder_hidden_states
        else:
            layer = int(len(outputs.hidden_states) * np.random.beta(self.alpha, self.beta))
            hs = torch.mean(outputs.hidden_states[layer], axis=1)
            # del outputs.hidden_states
        # torch.cuda.empty_cache()
        logits = outputs.logits 
        # get the error on the logits
        logits = torch.permute(logits, (0, 2, 1))
        logit_loss = torch.mean(F.cross_entropy(logits, labels, reduction="none"), axis=1)
        logit_loss = logit_loss.view(-1, 1)
        kd_score = self.calc_knowledge_discontinuities(logit_loss, hs)
        # torch.cuda.empty_cache()

        if return_outputs:
            return torch.sum(logit_loss) + self.lam * kd_score, outputs
        return torch.sum(logit_loss) + self.lam * kd_score


train_dataset, valid_dataset, test_dataset = preprocess_dataset("cnn_dailymail", tokenizer)

# train_dataset = train_dataset.select(range(10))
# valid_dataset = valid_dataset.select(range(10))

# peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)


if "gpt2" in options.model_name:
    model = GPT2LMHeadModel.from_pretrained(options.model_name)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(options.model_name)
    # model = get_peft_model(model, peft_config)
    # print(model.print_trainable_parameters())

training_args = Seq2SeqTrainingArguments(
    output_dir=f"sum-cnn{'-kd' if options.kd else ''}-flan-t5",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    # eval_accumulation_steps=2,
    learning_rate=options.learning_rate,
    num_train_epochs=options.epochs,
    evaluation_strategy="epoch",
    # eval_steps=2,
    predict_with_generate=True,
    fp16=True,
    weight_decay=options.weight_decay,
    hub_token=os.environ.get("HUB_TOKEN"),
    report_to="tensorboard",
    push_to_hub=True,
)
trainer = SummarizationTrainer(
    kd=options.kd,
    alpha=options.alpha,
    beta=options.beta,
    lam=options.lam,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)
trainer.train()
