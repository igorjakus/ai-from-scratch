import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda"
eval_iters = 200
n_embd = 384
n_layer = 6
n_head = 6
dropout_prob = 0.2
# -------------------

torch.manual_seed(42)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class LayerNorm(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        eps = 1e-5
        return self.gain * ((x - x.mean(dim=-1, keepdim=True)) * (x.var(dim=-1, keepdim=True) + eps)**-0.5) + self.bias


class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.Q = nn.Linear(n_embd, head_size, bias=False)
        self.K = nn.Linear(n_embd, head_size, bias=False)
        self.V = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        B, T, C = x.shape
        q = self.Q(x) # (B, T, head_size)
        k = self.K(x) # (B, T, head_size)
        v = self.V(x) # (B, T, head_size)
        head_size = q.shape[-1]
        
        wei = q @ k.transpose(-1, -2) * (head_size**-0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # so that the model doesn't always look at the same tokens
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        # x -> (B, T, C)
        # head(x) -> (B, T, head_size)
        # torch.cat(...) -> (B, T, num_heads * head_size)
        # so we want C = num_heads * head_size to preserve the dimensionality
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)  # exchanging information between heads
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_prob),  # residual dropout
        )
    def forward(self, x):
        # x -> (B, T, C)
        return self.net(x) # (B, T, C)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln1 = LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln2 = LayerNorm(n_embd)
    
    def forward(self, x):
        # x -> (B, T, C)
        x = self.sa(self.ln1(x)) + x # (B, T, C), residual connection
        x = self.ffwd(self.ln2(x)) + x # (B, T, C), residual connection
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(n_layer)])
        self.ln = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPTLanguageModel().to(device)
model.train()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

pbar = tqdm(range(max_iters))
for iter in pbar:
    if iter % eval_interval == 0:
        losses = estimate_loss()
        pbar.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
