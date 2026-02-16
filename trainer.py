class Trainer:
    def __init__(self, model, tokenizer, lr=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        print(f"[info] Trainer initialized with learning rate {lr}")

    def train_step(self, context, targets):
        print("[info] Starting training step")
        print("[info] Running forward pass")
        logits, probs = self.model.forward(context)
        print("[info] Computing loss")
        loss = self.model.loss(probs, targets)
        print("[info] Running backward pass")
        grads = self.model.backward(probs, targets, logits, context)
        print("[info] Updating model parameters")
        self.model.update(grads, self.lr)
        print(f"[info] Training step completed with loss={loss:.4f}")
        return loss

    def train(self, context, targets, epochs=100):
        print(f"[info] Starting training for {epochs} epochs")
        for epoch in range(epochs):
            print(f"[info] Epoch {epoch+1}/{epochs}")
            loss = self.train_step(context, targets)
            if epoch % 10 == 0:
                print(f"[status] Epoch {epoch}, loss={loss:.4f}")
