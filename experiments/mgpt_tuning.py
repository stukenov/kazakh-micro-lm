# Assuming transformers and datasets libraries are installed
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

model_name = "sberbank-ai/mGPT"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


# Загрузите датасет и ограничьте его первыми 100 элементами
data_files = {'train': 'kaz_instruction_shuffled.json'}
dataset = load_dataset('AmanMussa/kazakh-instruction-v1', data_files=data_files)
dataset = dataset['train'].select(range(100))  # Используйте только первые 100 элементов


def collate_batch(batch):
    tokenized_texts = [tokenizer.encode(d['text'], return_tensors='pt').squeeze(0) for d in batch]
    padded_texts = pad_sequence(tokenized_texts, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = padded_texts.clone()  # Use the input IDs as labels for language modeling
    return {'input_ids': padded_texts, 'labels': labels}


dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=collate_batch)

# Adjust the freezing logic if necessary
for n, p in model.named_parameters():
    if 'transformer.h' in n and 'ln_' not in n:
        layer_num = int(n.split('.')[2])
        if layer_num > 0 and layer_num < model.config.n_layer - 1:
            p.requires_grad = False

optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for epoch in range(3):
    print('Epoch', epoch)
    progressbar = tqdm(dataloader)
    for batch in progressbar:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        progressbar.set_description("Loss: %.3f" % np.mean(loss.item()))
