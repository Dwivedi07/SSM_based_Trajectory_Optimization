# train_mamba_llm.py
import os
import sys

import random

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
# print(root_folder)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt 

# Configuration parameters
EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a small dataset (we'll use wikitext-2 as an example)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenization


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token exists

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model configuration
config = MambaConfig(
    d_model=1012,
    n_layer=14,
    d_intermediate=0,
    vocab_size=tokenizer.vocab_size,
    ssm_cfg={"layer": "Mamba1"},
    attn_layer_idx=[],
    attn_cfg={},
    rms_norm=True,
    residual_in_fp32=False,
    fused_add_norm=True,
    pad_vocab_size_multiple=8,
    tie_embeddings=True
)

# Initialize model
model = MambaLMHeadModel(config)
print(model)
print(f"Number of parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad))/ 1e6:.2f}M")

model.to(DEVICE)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
train_loss_history = []  


# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
    for batch in progress_bar:
        input_ids = torch.stack(batch["input_ids"]).to(DEVICE)
        labels = input_ids.clone()

        optimizer.zero_grad()
        outputs = model(input_ids)
        logits = outputs.logits

        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        train_loss_history.append(loss.item())  

        progress_bar.set_postfix(loss=loss.item())
        

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training Loss over Batches")
plt.legend()
plt.grid()
plt.show()

# Save the model
model.save_pretrained("mamba_model")
tokenizer.save_pretrained("mamba_model")

