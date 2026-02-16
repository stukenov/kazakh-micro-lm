import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load additional tokens from bpe_tokenizer.json
with open("bpe_tokenizer.json", "r") as f:
    bpe_tokens = json.load(f)
    if isinstance(bpe_tokens, dict):
        bpe_tokens = list(bpe_tokens.keys())  # Convert dict keys to list
    tokenizer.add_tokens(bpe_tokens)

# Ensure that the tokenizer has a pad token, use the eos_token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Save the tokenizer and resize model embeddings to match the new tokenizer size
tokenizer.save_pretrained("tokenizer")
model.resize_token_embeddings(len(tokenizer))

# Save the model
model.save_pretrained("model-qaz")

# Save the config
config = model.config
config.save_pretrained("model-qaz")

# Save the vocab
tokenizer.save_vocabulary("model-qaz")

# Save the tokenizer
tokenizer.save_pretrained("model-qaz")

from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Prepare data for training
train_dataset = load_dataset("kz-transformers/multidomain-kazakh-dataset", data_files="leipzig.csv", split='train')
train_dataset = train_dataset.filter(lambda e: e['predicted_language'] == 'kaz')
train_dataset = train_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Define training parameters
training_args = TrainingArguments(
    output_dir="./model-qaz", # where to save the model
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=16, # batch size for training
    warmup_steps=500, # warmup steps
    weight_decay=0.01, # weight decay
    logging_dir='./logs', # where to save logs
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model, # the model to be trained
    args=training_args, # training arguments
    train_dataset=train_dataset, # training dataset
)

# Start training
trainer.train()
