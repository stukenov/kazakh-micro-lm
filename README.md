# Kazakh Micro Language Model

A minimal, educational implementation of language models specifically designed for Kazakh text. This project demonstrates how to build neural language models from scratch using both pure Python and PyTorch implementations.

## Overview

**kazakh-micro-lm** is a lightweight language model implementation that showcases the fundamentals of modern language modeling techniques. It includes two different approaches:

1. **Pure Python Implementation** - A minimal RNN-style model with no dependencies beyond standard library
2. **PyTorch GPT-2 Implementation** - A small-scale GPT-2 architecture optimized for Kazakh text

Perfect for educational purposes, experimentation, and understanding the core concepts behind modern language models.

## Features

- **Zero-Dependency Core**: Pure Python implementation requires no external libraries
- **Kazakh Language Support**: Pre-trained tokenizer optimized for Kazakh text
- **Educational Focus**: Clear, readable code with extensive logging
- **Multiple Implementations**: Learn from both simple and advanced architectures
- **Easy to Extend**: Modular design makes it simple to experiment with new ideas

## Quick Start

### Pure Python Implementation

```bash
# Train a model on your text
python main.py --mode train --input abay-joly.txt --epochs 100 --lr 0.1

# Generate text
python main.py --mode infer --prompt "шла саша" --max_len 10
```

### PyTorch GPT-2 Implementation

```bash
# Install dependencies
pip install torch transformers tqdm requests

# Train the model
python gpt2_minimal_kazakh.py
```

## Project Structure

```
kazakh-micro-lm/
├── model.py                  # Pure Python RNN implementation
├── trainer.py                # Training loop for pure Python model
├── inference.py              # Text generation utilities
├── tokenizer.py              # Simple word-level tokenizer
├── main.py                   # CLI interface for training/inference
├── gpt2_minimal_kazakh.py    # PyTorch GPT-2 implementation
├── gpt2.c                    # C implementation (experimental)
├── abay-joly.txt             # Sample Kazakh text for training
├── vocab.json                # Pre-trained vocabulary
├── merges.txt                # BPE merges for tokenization
├── tokenizer.json            # Tokenizer configuration
└── model.json                # Saved model weights
```

## Implementation Details

### Pure Python Model

The pure Python implementation features:
- Simple embedding layer with one-hot encoding
- Matrix operations implemented from scratch
- Gradient descent optimization
- Cross-entropy loss
- JSON-based model serialization

### PyTorch GPT-2 Model

The PyTorch implementation includes:
- Multi-head self-attention mechanism
- Transformer blocks with layer normalization
- Causal language modeling objective
- Mixed precision training support
- CUDA acceleration when available

## Training Your Own Model

### Using Custom Text

1. Prepare your Kazakh text file (UTF-8 encoded)
2. Train with the pure Python implementation:

```bash
python main.py --mode train --input your_text.txt --epochs 200 --lr 0.05
```

3. Or use the PyTorch implementation for better results:

```bash
python gpt2_minimal_kazakh.py
# Modify the script to point to your training data
```

### Hyperparameters

**Pure Python Model:**
- `--epochs`: Number of training iterations (default: 100)
- `--lr`: Learning rate (default: 0.1)
- `--max_len`: Maximum generation length (default: 5)

**PyTorch GPT-2 Model:**
- `ctx_len`: Context length (default: 128)
- `n_emb`: Embedding dimension (default: 256)
- `n_heads`: Number of attention heads (default: 8)
- `n_layers`: Number of transformer layers (default: 6)
- `batch_size`: Training batch size (default: 64)

## Use Cases

- **Education**: Learn how language models work under the hood
- **Research**: Experiment with new architectures and training techniques
- **Prototyping**: Quickly test ideas before scaling up
- **Kazakh NLP**: Build applications for Kazakh language processing
- **Low-Resource Settings**: Train models without heavy computational requirements

## Requirements

### Pure Python Implementation
- Python 3.6+
- No external dependencies

### PyTorch Implementation
- Python 3.8+
- PyTorch 2.0+
- transformers
- tqdm
- requests

## Performance Notes

The pure Python implementation is intentionally minimal and not optimized for speed. It's designed for educational purposes to understand the fundamentals. For production use cases, the PyTorch implementation offers:

- GPU acceleration (10-100x faster)
- Mixed precision training
- Efficient batch processing
- Modern attention mechanisms

## Contributing

Contributions are welcome! This project is designed to be simple and educational, so please keep changes minimal and well-documented. Areas for contribution:

- Additional training examples
- Performance optimizations
- Better documentation
- Support for other Turkic languages
- Evaluation metrics and benchmarks

## License

MIT License - see LICENSE file for details

## Citation

If you use this project in your research or education, please cite:

```bibtex
@software{kazakh_micro_lm,
  title = {Kazakh Micro Language Model},
  author = {Stukenov, Saken},
  year = {2024},
  url = {https://github.com/stukenov/kazakh-micro-lm}
}
```

## Acknowledgments

- Inspired by Andrej Karpathy's educational materials on neural networks
- Tokenizer trained on Kazakh text corpus
- Built with educational simplicity in mind

## Related Projects

- [minGPT](https://github.com/karpathy/minGPT) - Minimal PyTorch re-implementation of GPT
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Simplest, fastest repository for training GPT
- [micrograd](https://github.com/karpathy/micrograd) - Tiny autograd engine

---

Made with focus on education and simplicity for the Kazakh language community.
