import json
import sys

class Tokenizer:
    def __init__(self):
        sys.stderr.write("info: initializing tokenizer\n")
        self.word_to_ix = {}
        self.ix_to_word = {}
        self.vocab_size = 0

    def fit(self, text):
        sys.stderr.write("info: fitting tokenizer on text\n")
        tokens = text.split()
        vocab = list(set(tokens))
        self.vocab_size = len(vocab)
        self.word_to_ix = {w:i for i,w in enumerate(vocab)}
        self.ix_to_word = {i:w for w,i in self.word_to_ix.items()}
        sys.stderr.write(f"info: built vocabulary of size {self.vocab_size}\n")

    def encode(self, text):
        sys.stderr.write("info: encoding text\n")
        tokens = text.split()
        return [self.word_to_ix[t] for t in tokens]

    def decode(self, indices):
        sys.stderr.write("info: decoding indices\n") 
        return ' '.join([self.ix_to_word[i] for i in indices])

    def save(self, filename):
        sys.stderr.write(f"info: saving tokenizer to {filename}\n")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'word_to_ix': self.word_to_ix
            }, f)

    def load(self, filename):
        sys.stderr.write(f"info: loading tokenizer from {filename}\n")
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.word_to_ix = data['word_to_ix']
            self.ix_to_word = {i:w for w,i in self.word_to_ix.items()}
            self.vocab_size = len(self.word_to_ix)
            sys.stderr.write(f"info: loaded vocabulary of size {self.vocab_size}\n")
