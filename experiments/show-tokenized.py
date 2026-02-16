import os
import numpy as np
import tiktoken

# Load the encoded train data from the binary file
train_ids_path = os.path.join(os.path.dirname(__file__), 'train.bin')
train_ids = np.fromfile(train_ids_path, dtype=np.uint16)

# Decode the first 10 train IDs to their corresponding text
enc = tiktoken.get_encoding("gpt2")
decoded_tokens = enc.decode(train_ids[:100])
print("Decoded text from the first 10 train IDs:")
print(decoded_tokens)

# Show example use
