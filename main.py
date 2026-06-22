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

        # --- Block 1: Causal Self-Attention ---
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

        # --- Block 2: MLP Block ---
        x = x + self.mlp(self.ln2(x))

        # Output Logits
        logits = x @ self.Wout
        return logits


# ==========================================
# 3. 測試執行區塊（這是原本缺少的啟動開關！）
# ==========================================
if __name__ == "__main__":
    print("====== 正在初始化 GPT 2.0 模型 ======")
    
    # 準備測試文本
    sample_text = "Hello GPT! This is a simple language model implementation text."
    
    # 1. 初始化 Tokenizer
    tokenizer = SimpleTokenizer(sample_text)
    print(f"成功建立詞表！詞表大小 (Vocab Size): {tokenizer.vocab_size}")
    
    # 2. 設定模型參數並建立模型
    embed_dim = 32
    seq_len = 16
    model = CustomGPT(vocab_size=tokenizer.vocab_size, embed_dim=embed_dim, seq_len=seq_len)
    
    # 3. 將測試文字轉成數字矩陣 (Tensor) 餵給模型
    test_input = "Hello GPT!"
    encoded_input = tokenizer.encode(test_input)
    # 轉成 PyTorch 需要的維度 (Batch_size=1, Sequence_length)
    input_tensor = torch.tensor([encoded_input], dtype=torch.long)
    
    print(f"輸入文字: '{test_input}' -> 編碼後的 Token ID: {encoded_input}")
    
    # 4. 前向傳播 (Forward Pass) 測試
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print("=== 模型執行成功！ ===")
    print(f"模型輸出形狀 (Batch, Length, Vocab): {outputs.shape}")
    print("模型已經可以正常運作與計算了！")
