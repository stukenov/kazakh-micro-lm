# -*- coding: utf-8 -*-
"""
GPT-2 minimal implementation for Kazakh language
Adapted from Colab notebook to run as a standalone Python script
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import math
import os
import logging
import random
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from tqdm import tqdm
import pickle
import requests
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Enable cuDNN benchmark for optimized performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logger.info("cuDNN benchmark enabled.")

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Download tokenizer files if they don't exist
def download_file(url, filename):
    print(f"Downloading {filename} from {url}")
    response = requests.get(url, stream=True)
    total_length = response.headers.get('content-length')

    if total_length is None:  # no content length header
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        with open(filename, 'wb') as f:
            dl = 0
            total_length = int(total_length)
            for data_chunk in tqdm(
                response.iter_content(chunk_size=1024*1024), 
                total=math.ceil(total_length/(1024*1024)), 
                desc=f"Downloading {filename}", 
                unit="MB"
            ):
                dl += len(data_chunk)
                f.write(data_chunk)
    print(f"Downloaded {filename}")

if not os.path.exists("vocab.json"):
    download_file(
        "https://github.com/sakentsunofu/data/raw/refs/heads/main/vocab.json", 
        "vocab.json"
    )
if not os.path.exists("merges.txt"):
    download_file(
        "https://github.com/sakentsunofu/data/raw/refs/heads/main/merges.txt", 
        "merges.txt"
    )

# Initialize tokenizer without adding a new pad token
tokenizer = GPT2Tokenizer(
    vocab_file="./vocab.json",
    merges_file="./merges.txt",
    bos_token='<s>',
    eos_token='</s>',
    pad_token='</s>'  # Set pad_token to eos_token to avoid adding a new token
)

vocab_size = tokenizer.vocab_size  # This should remain consistent with the model
print(f"Tokenizer vocab_size = {vocab_size}")

# Initialize fast tokenizer with the same pad_token
tokenizer_fast = GPT2TokenizerFast(
    vocab_file="./vocab.json",
    merges_file="./merges.txt",
    bos_token='<s>',
    eos_token='</s>',
    pad_token='</s>'  # Ensure pad_token consistency
)

# Load preprocessed training data
preprocessed_dir = "/content/preprocessed_data"
os.makedirs(preprocessed_dir, exist_ok=True)

train_preprocessed_path = os.path.join(preprocessed_dir, "train_tokenized.pkl")

if not os.path.exists(train_preprocessed_path):
    download_file(
        "https://huggingface.co/saken-tukenov/gpt2-nano/resolve/main/train_tokenized.pkl", 
        train_preprocessed_path
    )

print("Loading training data...")
with open(train_preprocessed_path, 'rb') as f:
    train_tokenized = pickle.load(f)
print(f"Loaded training data from {train_preprocessed_path}")

# Convert to torch.Tensor for faster indexing
# Handle different possible data formats
if isinstance(train_tokenized, list):
    try:
        train_tokenized = torch.tensor(train_tokenized, dtype=torch.long)
        print("Converted training data to torch.Tensor.")
    except Exception as e:
        raise TypeError(f"Error converting list to tensor: {e}")
elif isinstance(train_tokenized, np.ndarray):
    train_tokenized = torch.from_numpy(train_tokenized).long()
    print("Converted numpy array to torch.Tensor.")
elif isinstance(train_tokenized, torch.Tensor):
    train_tokenized = train_tokenized.long()
    print("Training data is already a torch.Tensor. Converted to long dtype.")
else:
    raise TypeError("train_tokenized must be a list, numpy array, or torch.Tensor of integers.")

# Validate token ids to ensure they are within the vocabulary range
max_token_id = train_tokenized.max().item()
if max_token_id >= vocab_size:
    raise ValueError(f"Token ID {max_token_id} exceeds the vocabulary size of {vocab_size}.")

# Model hyperparameters
ctx_len = 128     # Context length for training/generation
n_emb = 256       # Increased Embedding size for better performance
dropout = 0.1
head_size = 256
n_heads = 8
n_layers = 6       # Increased number of layers for better learning

# Training hyperparameters
num_epochs = 10    # Increased epochs for better convergence
batch_size = 64    # Increased batch size for faster training
lr = 5e-4          # Adjusted learning rate
weight_decay = 1e-1 # Added weight decay for regularization

class TextDataset(data.Dataset):
    def __init__(self, tokenized_texts, block_size=128):
        self.tokenized_texts = tokenized_texts
        self.block_size = block_size

    def __len__(self):
        return self.tokenized_texts.size(0)

    def __getitem__(self, idx):
        input_ids = self.tokenized_texts[idx]
        x = input_ids[:self.block_size]
        y = input_ids[1:self.block_size+1]
        return x, y

# Create dataset
print("Creating training dataset...")
train_dataset = TextDataset(train_tokenized, block_size=ctx_len)
print(f"Train dataset size: {len(train_dataset)}")

# DataLoader with optimizations
num_workers = os.cpu_count() if os.cpu_count() is not None else 4
pin_memory = True if device == 'cuda' else False

train_loader = data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=True,  # Keeps workers alive between epochs
    prefetch_factor=4  # Number of batches loaded in advance by each worker
)

def get_batches_from_loader(loader):
    for batch in loader:
        input_, label_ = batch
        yield input_.to(device, non_blocking=True), label_.to(device, non_blocking=True)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(n_emb, head_size * n_heads, bias=False)
        self.q_proj = nn.Linear(n_emb, head_size * n_heads, bias=False)
        self.v_proj = nn.Linear(n_emb, head_size * n_heads, bias=False)

        # Ensure indices are created on the correct device
        indices = torch.arange(ctx_len, device=device).unsqueeze(1) < torch.arange(ctx_len, device=device).unsqueeze(0)
        self.register_buffer(
            '_causal_mask',
            torch.where(indices, torch.tensor(0.0, device=device), torch.tensor(-1e9, device=device))
        )

        self.c_proj = nn.Linear(head_size * n_heads, n_emb)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        K = self.k_proj(x)
        Q = self.q_proj(x)
        V = self.v_proj(x)

        head_dim = head_size
        # Reshape for multi-head
        K = K.view(B, T, n_heads, head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        Q = Q.view(B, T, n_heads, head_dim).transpose(1, 2)
        V = V.view(B, T, n_heads, head_dim).transpose(1, 2)

        attn_weights = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = attn_weights + self._causal_mask[:T, :T]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        o = attn_weights @ V
        o = o.transpose(1, 2).contiguous().view(B, T, n_heads * head_dim)
        o = self.c_proj(self.resid_dropout(o))
        return o

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_emb, 4 * n_emb)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_emb)
        self.mha = MultiHeadAttention()
        self.ln_2 = nn.LayerNorm(n_emb)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_emb, padding_idx=tokenizer.eos_token_id)
        self.wpe = nn.Embedding(ctx_len, n_emb)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'c_proj' in name:
                    nn.init.normal_(module.weight, mean=0.0, std=(0.02 / math.sqrt(2 * n_layers)))
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids=None, max_new_tokens=100, temperature=1.0, top_k=None):
        if prompt_ids is None:
            ctx = torch.zeros((1, 1), dtype=torch.long, device=next(self.parameters()).device)
        else:
            ctx = torch.tensor(prompt_ids, dtype=torch.long, device=next(self.parameters()).device).unsqueeze(0)
        for _ in range(max_new_tokens):
            idx_cond = ctx[:, -ctx_len:] if ctx.size(1) > ctx_len else ctx
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                threshold = v[:, -1].unsqueeze(-1)
                logits[logits < threshold] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ctx = torch.cat((ctx, next_token), dim=1)
        return ctx[0].tolist()

def loss_fn(model, x, y):
    logits = model(x)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    y = y.view(B*T)
    loss = nn.functional.cross_entropy(logits, y, reduction='mean')
    return loss

def main():
    print("Initializing model and optimizer...")
    # Initialize model and optimizer
    model = GPT().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr, 
        steps_per_epoch=len(train_loader), 
        epochs=num_epochs,
        pct_start=0.1,
        anneal_strategy='linear'
    )

    # Try to compile model
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    except Exception as e:
        print(f"Using model without torch.compile(). Reason: {e}")

    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler(enabled=(device=='cuda'))

    print("Starting training loop...")
    # Training loop
    total_batches = len(train_loader) * num_epochs
    current_batch = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} - Training Phase")
        # Training phase
        model.train()
        running_loss = 0.0
        batch_cnt = 0

        for input_, label_ in tqdm(
            get_batches_from_loader(train_loader), 
            desc=f"Training Batches ({current_batch}/{total_batches})", 
            unit="batch"
        ):
            current_batch += 1
            batch_cnt += 1
            # input_ and label_ are already moved to device in get_batches_from_loader

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device, enabled=(device=='cuda')):
                loss = loss_fn(model, input_, label_)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / batch_cnt
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    # Generate sample text
    print("Generating sample text...")
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(max_new_tokens=100, temperature=1.0, top_k=50)
    completion_text = tokenizer_fast.decode(generated_ids)

    print("=== Generated text ===")
    print(completion_text)

    # Save model
    print("Saving model to model.pt")
    torch.save(model.state_dict(), "model.pt")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()