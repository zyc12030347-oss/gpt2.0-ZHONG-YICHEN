# gpt2.0-ZHONG-YICHEN
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. SimpleTokenizer
# ==========================================
class SimpleTokenizer:
    
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])


# ==========================================
# 2. CustomGPT)
# ==========================================
class CustomGPT(nn.Module):

    def __init__(self, vocab_size, embed_dim, seq_len):
        super().__init__()
        # 1. 
        self.embeddings = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)
        self.pos_embeddings = nn.Parameter(torch.randn(seq_len, embed_dim) * 0.02)

        # 2. Pre-LN
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # 3. 
        self.Wq = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.Wk = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.Wv = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)

        # 4. MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

        # 5. 
        self.Wout = nn.Parameter(torch.randn(embed_dim, vocab_size) * 0.02)

    def forward(self, idx):
        B, T = idx.shape
        
    
        x = self.embeddings[idx] + self.pos_embeddings[:T]

        # ---1---
        x_norm = self.ln1(x)
        q = x_norm @ self.Wq
        k = x_norm @ self.Wk
        v = x_norm @ self.Wv

        # $Score = \frac{QK^T}{\sqrt{d_k}}$
        attn_scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        
        # Causal Mask
        mask = torch.tril(torch.ones(T, T, device=idx.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = attn_probs @ v
        
        # 
        x = x + attn_out

        # ---2---
        x = x + self.mlp(self.ln2(x))

        # Logits
        logits = x @ self.Wout
        return logits


# ==========================================
# 3. Inference
# ==========================================
def generate(model, tokenizer, start_char, max_new_tokens=30, device='cpu'):

    model.eval()
    idx = torch.tensor([[tokenizer.stoi[start_char]]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            
            max_seq_len = model.pos_embeddings.size(0)
            idx_cond = idx[:, -max_seq_len:]
            
            logits = model(idx_cond)
           
            probs = F.softmax(logits[:, -1, :], dim=-1)
            
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
            
    return tokenizer.decode(idx[0].tolist())


# ==========================================
# 4. Main Pipeline
# ==========================================
if __name__ == "__main__":
    # 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # data.txt
    if not os.path.exists('data.txt'):
        mock_data = "hello world! welcome to the tiny gpt implementation. python and pytorch are awesome."
        with open('data.txt', 'w', encoding='utf-8') as f:
            f.write(mock_data)
        print("Generated a temporary 'data.txt' for training.")

    # 1. 
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. 
    tokenizer = SimpleTokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)

    # Batch_size = 1
    x = tokens[:-1].unsqueeze(0)  # 形状: (1, T)
    y = tokens[1:].unsqueeze(0)   # 形状: (1, T)

    # 3. 
    model = CustomGPT(vocab_size=tokenizer.vocab_size, embed_dim=64, seq_len=x.size(1))
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

    # 4. 
    print("시작...")
    for step in range(301):
        model.train()
        logits = model(x)  #  (B, T, vocab_size)
        
        # 
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")

    # 5. 
    print("\n모델 생성 결과 테스트:")
    start_char = text[0]
    generated_text = generate(model, tokenizer, start_char=start_char, max_new_tokens=40, device=device)
    print(f"입력: '{start_char}' -> 텍스트 생성: \n{generated_text}")
