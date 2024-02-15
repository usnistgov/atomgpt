from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from tqdm import tqdm
import transformers
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import os

os.environ["WANDB_ANONYMOUS"] = "must"
IGNORE_INDEX = -100

# Define a custom dataset class for regression
class AtomGPTDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return inputs, torch.tensor(self.targets[idx], dtype=torch.float32)


# Example usage
if __name__ == "__main__":

    # Load pre-trained tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model_name = "gpt2"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    batch_size = 2
    max_length = 128
    num_epochs = 3
    learning_rate = 5e-5
    # Define example regression data (texts and corresponding numeric targets)
    train_texts = [
        "This is the first example text.",
        "Second example is a bit longer than the first one, but still within the max length.",
        "Third example is the longest among these three examples. It exceeds the max length and will be truncated.",
        "Second example is a bit longer than the first one, but still within the max length.",
    ]
    train_targets = [10.2, 15.5, 20.1, 15.5]  # Example regression targets

    # Fine-tune the last layer of GPT-2 for regression
    # fine_tune_gpt2_regression(train_texts, train_targets, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.lm_head = torch.nn.Linear(
        model.config.hidden_size, 1
    )  # Single output for regression

    # Prepare datasets and dataloaders with data collator
    train_dataset = AtomGPTDataset(
        train_texts, train_targets, tokenizer, max_length=max_length
    )
    val_dataset = train_dataset

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(num_epochs):
        model.train()
        # print("train_dataloader",train_dataloader,len(train_dataloader),train_dataloader[0])
        for batch in train_dataloader:
            print(batch, type(batch))
            # predictions = model(batch_inputs, batch_masks)
            input_ids = batch[0]["input_ids"].squeeze() #.squeeze(0)
            #input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        
            #print("input_ids",input_ids.shape,input_ids.squeeze().shape)
            #pred = model(input_ids)#,batch[0]["attention_mask"]) #.logits.squeeze().mean(dim=1)
            pred = model(input_ids).logits.squeeze().mean(dim=1)
            print("pred", pred, len(pred))
            break
