class Inference:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        print("[info] Inference initialized with model and tokenizer")

    def generate(self, prompt, max_len=5):
        print(f"[info] Starting generation with prompt: {prompt}")
        context = self.tokenizer.encode(prompt)
        print(f"[info] Encoded prompt to context tokens: {context}")
        
        for i in range(max_len):
            print(f"[info] Generation step {i+1}/{max_len}")
            logits, probs = self.model.forward(context)
            # Берем вероятности для последнего токена
            last_probs = probs[-1]
            # Выбираем токен с максимальной вероятностью
            next_ix = last_probs.index(max(last_probs))
            print(f"[info] Selected next token with index: {next_ix}")
            context.append(next_ix)
            
        result = self.tokenizer.decode(context)
        print(f"[info] Generation complete. Result: {result}")
        return result
