def get_vocab(text):
    words = text.split()
    return {word: words.count(word) for word in set(words)}

def get_stats(vocab):
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs

def merge_vocab(pair, vocab):
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    return {word.replace(bigram, replacement): freq for word, freq in vocab.items()}

def bpe_iterations(text, num_merges):
    vocab = get_vocab(text)
    vocab = {' '.join(word): freq for word, freq in vocab.items()}
    print("Initial vocabulary:", vocab)

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        print(f"After iteration {i + 1}, Best pair: {best_pair}")
        print("Updated vocabulary:", vocab)
    return vocab

text = open('input.txt', 'r', encoding='utf-8').read()
num_merges = 10
final_vocab = bpe_iterations(text, num_merges)
print("Final vocabulary:", final_vocab)
