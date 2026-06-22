import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. SimpleTokenizer
# ==========================================
class SimpleTokenizer:
    """A simple character-level tokenizer for toy language model training."""
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
# 2. CustomGPT
# ==========================================
class CustomGPT(nn.Module):
    """A minimal Pre-LN causal Transformer (GPT) model architecture."""
    def __init__(self, vocab_size, embed_dim, seq_len):
        super().__init__()
        # 1. Embedding Layers
        self.embeddings = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)
        self.pos_embeddings = nn.Parameter(torch.randn(seq_len, embed_dim) * 0.02)

        # 2. Layer Normalization (Pre-LN Architecture)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # 3. Attention Projection Matrices
        self.Wq = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.Wk = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.Wv = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)

        # 4. Multi-Layer Perceptron (MLP Block)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

        # 5. Output Projection Layer
        self.Wout = nn.Parameter(torch.randn(embed_dim, vocab_size) * 0.02)

    def forward(self, idx):
        B, T = idx.shape
        
        # Token embedding + absolute position encoding
        x = self.embeddings[idx] + self.pos_embeddings[:T]

        # --- Block 1: Causal Self-Attention (with Residual Connection & LayerNorm) ---
        x_norm = self.ln1(x)
        q = x_norm @ self.Wq
        k = x_norm @ self.Wk
        v = x_norm @ self.Wv

        # Calculate scaled dot-product attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        
        # Apply Causal Mask
        mask = torch.tril(torch.ones(T, T, device=idx.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = attn_probs @ v
        
        # Residual connection 1
        x = x + attn_out
