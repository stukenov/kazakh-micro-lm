import math
import json
import random
import sys
import time
from datetime import datetime

def zeros(shape):
    sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating zero matrix of shape {shape}\n")
    return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]

def rand_matrix(rows, cols, scale=0.01):
    sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating random matrix {rows}x{cols} with scale {scale}\n")
    return [[(random.random()*2-1)*scale for _ in range(cols)] for _ in range(rows)]

def matmul(a, b):
    M = len(a)
    N = len(a[0])
    K = len(b[0])
    sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Matrix multiplication {M}x{N} @ {N}x{K}\n")
    out = zeros((M,K))
    for i in range(M):
        for j in range(K):
            s = 0.0
            for k in range(N):
                s += a[i][k]*b[k][j]
            out[i][j] = s
    return out

def add(a, b):
    M = len(a)
    N = len(a[0])
    sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Matrix addition {M}x{N}\n")
    if len(b) == 1 and len(b[0]) == N:
        out = zeros((M, N))
        for i in range(M):
            for j in range(N):
                out[i][j] = a[i][j] + b[0][j]
        return out
    else:
        out = zeros((M,N))
        for i in range(M):
            for j in range(N):
                out[i][j] = a[i][j] + b[i][j]
        return out

def softmax(a):
    max_val = max(a[0])
    exps = [math.exp(v - max_val) for v in a[0]]
    sum_exps = sum(exps)
    return [[e/sum_exps for e in exps]]

class GPTModel:
    def __init__(self, vocab_size, d_model=64):
        sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing GPT model with vocab_size={vocab_size}, d_model={d_model}\n")
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Пример: эмбеддинг и выходной слой
        self.W_emb = rand_matrix(vocab_size, d_model)
        self.b_emb = zeros((1,d_model))
        self.W_out = rand_matrix(d_model, vocab_size)
        self.b_out = zeros((1,vocab_size))

    def forward(self, indices):
        sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Forward pass with sequence length {len(indices)}\n")
        T = len(indices)
        X_onehot = zeros((T, self.vocab_size))
        for i in range(T):
            X_onehot[i][indices[i]] = 1.0
        # Embedding
        X_emb = add(matmul(X_onehot,self.W_emb), self.b_emb)
        # Вывод
        logits = add(matmul(X_emb, self.W_out), self.b_out)
        probs = []
        for i in range(T):
            p = softmax([logits[i]])
            probs.append(p[0])
        return logits, probs

    def loss(self, probs, targets):
        sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Computing loss\n")
        T = len(probs)
        loss = 0.0
        for i in range(T):
            correct_p = probs[i][targets[i]]
            loss -= math.log(correct_p+1e-9)
        loss /= T
        return loss

    def backward(self, probs, targets, logits, indices):
        sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Backward pass\n")
        T = len(probs)
        dProbs = zeros((T,self.vocab_size))
        for i in range(T):
            for j in range(self.vocab_size):
                dProbs[i][j] = probs[i][j]
            dProbs[i][targets[i]] -= 1.0
        for i in range(T):
            for j in range(self.vocab_size):
                dProbs[i][j] /= T

        dLogits = dProbs
        dX_emb = matmul(dLogits, transpose(self.W_out))
        dW_out = matmul(transpose(dX_emb), dLogits)
        db_out = zeros((1,self.vocab_size))
        for i_ in range(T):
            for j_ in range(self.vocab_size):
                db_out[0][j_] += dLogits[i_][j_]

        X_onehot = zeros((T, self.vocab_size))
        for i_ in range(T):
            X_onehot[i_][indices[i_]] = 1.0
        dW_emb = matmul(transpose(X_onehot), dX_emb)
        db_emb = zeros((1,self.d_model))
        for i_ in range(T):
            for j_ in range(self.d_model):
                db_emb[0][j_] += dX_emb[i_][j_]

        grads = {
            'W_emb': dW_emb,
            'b_emb': db_emb,
            'W_out': dW_out,
            'b_out': db_out
        }
        return grads

    def update(self, grads, lr=0.1):
        sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating parameters with learning rate {lr}\n")
        dW_emb = grads['W_emb']
        db_emb = grads['b_emb']
        dW_out = grads['W_out']
        db_out = grads['b_out']

        for i in range(self.vocab_size):
            for j in range(self.d_model):
                self.W_emb[i][j] -= lr * dW_emb[i][j]
        for j in range(self.d_model):
            self.b_emb[0][j] -= lr * db_emb[0][j]

        for i in range(self.d_model):
            for j in range(self.vocab_size):
                self.W_out[i][j] -= lr * dW_out[i][j]
        for j in range(self.vocab_size):
            self.b_out[0][j] -= lr * db_out[0][j]

    def save(self, filename):
        sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving model to {filename}\n")
        data = {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'W_emb': self.W_emb,
            'b_emb': self.b_emb,
            'W_out': self.W_out,
            'b_out': self.b_out
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load(self, filename):
        sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading model from {filename}\n")
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab_size = data['vocab_size']
            self.d_model = data['d_model']
            self.W_emb = data['W_emb']
            self.b_emb = data['b_emb']
            self.W_out = data['W_out']
            self.b_out = data['b_out']

def transpose(a):
    M = len(a)
    N = len(a[0])
    sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Transposing matrix {M}x{N}\n")
    out = zeros((N,M))
    for i in range(M):
        for j in range(N):
            out[j][i] = a[i][j]
    return out
