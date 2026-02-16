import argparse
import sys
import time
from tokenizer import Tokenizer
from model import GPTModel
from trainer import Trainer
from inference import Inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple GPT-like model")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], 
                        help='Operation mode: train or infer')
    parser.add_argument('--input', type=str, default=None, 
                        help='Input text file for training')
    parser.add_argument('--model', type=str, default='model.json', 
                        help='Path to save/load model weights')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.json', 
                        help='Path to save/load tokenizer')
    parser.add_argument('--prompt', type=str, default="hello", 
                        help='Prompt for inference')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='Learning rate for training')
    parser.add_argument('--max_len', type=int, default=5,
                        help='Max length of generation for inference')

    args = parser.parse_args()

    if args.mode == 'train':
        if args.input is None:
            print("[ERROR] Please provide --input file for training mode.", file=sys.stderr)
            sys.exit(1)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting training mode")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Reading input file: {args.input}")
        # Читаем текст из файла
        with open(args.input, 'r', encoding='utf-8') as f:
            data = f.read()

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing tokenizer")
        # Инициализация токенайзера и обучение на тексте
        tokenizer = Tokenizer()
        tokenizer.fit(data)
        indices = tokenizer.encode(data)
        context = indices[:-1]
        targets = indices[1:]

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model with vocab_size={tokenizer.vocab_size}")
        # Инициализация и обучение модели
        model = GPTModel(vocab_size=tokenizer.vocab_size, d_model=8)
        trainer = Trainer(model, tokenizer, lr=args.lr)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting training for {args.epochs} epochs")
        trainer.train(context, targets, epochs=args.epochs)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving model to: {args.model}")
        # Сохраняем модель и токенайзер
        model.save(args.model)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving tokenizer to: {args.tokenizer}")
        tokenizer.save(args.tokenizer)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training completed successfully")

    elif args.mode == 'infer':
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting inference mode")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading tokenizer from: {args.tokenizer}")
        # Загрузка модели и токенайзера
        tokenizer = Tokenizer()
        tokenizer.load(args.tokenizer)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading model from: {args.model}")
        model = GPTModel(vocab_size=tokenizer.vocab_size, d_model=8)
        model.load(args.model)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating text for prompt: {args.prompt}")
        inference = Inference(model, tokenizer)
        generated = inference.generate(args.prompt, max_len=args.max_len)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generated text: {generated}")
