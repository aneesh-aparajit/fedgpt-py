from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dataset import vocab

# ---------------------------------- Config ---------------------------------- #
@dataclass
class GptConfig:
    buffer_size: int = 512
    vocab_size: int = len(vocab)  # GPT2 has a total of 50257, padded to nearest multiple of 64 for efficiency
    n_layers: int = 6
    n_head: int = 8
    n_embed: int = 768
    dropout: float = 0.1
    bias: bool = False
    use_sinusoidal: bool = True


# ----------------------------- Attention Module ----------------------------- #
class Attention(nn.Module):
    '''Unlike RNNs where we were required to get one output and then pass it back onto the RNN and repeat the process
    again and again, here with masked attention, we simply find the lower triangular matrix and then weight them according 
    the vector product the lower triangular matrix and the embedding vectors, we  build a masked representation for each word only using 
    the values which occured/was predicted prior to the current index. 
        - This is achieved by the torch.tril function and masking all zeros to -torch.inf (taking softmax makes it equal to zero)
    '''
    def __init__(self, n_embed: int, head_size: int) -> None:
        super().__init__()
        self.Q = nn.Linear(n_embed, head_size, bias=GptConfig.bias)
        self.K = nn.Linear(n_embed, head_size, bias=GptConfig.bias)
        self.V = nn.Linear(n_embed, head_size, bias=GptConfig.bias)
        tril = torch.tril(
            torch.ones(size=(GptConfig.buffer_size, GptConfig.buffer_size))
        )
        self.register_buffer("tril", tril)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        q, k, v = self.Q(x), self.K(x), self.V(x)  # (B, T, C) => (B, T, H)
        wei = (
            q @ k.mT * (1.0 / math.sqrt(k.size(-1)))
        )  # (B, T, H) @ (B, H, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v  # (B, T, T) @ (B, T, H) => (B, T, H)


# --------------------------- Multi Head Attention --------------------------- #
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_heads: int) -> None:
        super().__init__()
        assert (
            n_embed % n_heads == 0
        ), "The number of heads must divide the embedding dimensions"
        head_size = n_embed // n_heads
        self.heads = nn.ModuleList(
            [Attention(n_embed=n_embed, head_size=head_size) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed, bias=GptConfig.bias)
        self.dropout = nn.Dropout(p=GptConfig.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C) -> (B, T, C//N_HEADS) -> (B, T, C)
        return self.dropout(self.proj(x)) #  (B, T, C) -> (B, T, C)


# ------------------------------- Feed Forward ------------------------------- #
class FeedForward(nn.Module):
    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.GELU(), nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------ Attention Block ----------------------------- #
class AttentionBlock(nn.Module):
    def __init__(self, n_embed: int, n_heads: int) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(n_embed=n_embed, n_heads=n_heads)
        self.ffwd = FeedForward(n_embed=n_embed)
        self.ln1 = nn.LayerNorm((n_embed,))
        self.ln2 = nn.LayerNorm((n_embed,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x)) # (B, T, C) -> (B, T, C)
        x = x + self.ffwd(self.ln2(x))
        return x


# --------------------------- Positional Embeddings -------------------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, n_embed: int, max_seq_len: int) -> None:
        super().__init__()

        position_id = torch.arange(0, max_seq_len).unsqueeze(1)
        frequencies = torch.arange(0, n_embed, 2, dtype=torch.float32) / n_embed
        frequencies = torch.pow(10000.0, -frequencies)

        positional_encodings = torch.zeros(size=(max_seq_len, n_embed))
        # print(frequencies.shape, position_id.shape, positional_encodings.shape)

        positional_encodings[:, 0::2] = torch.sin(position_id * frequencies)
        positional_encodings[:, 1::2] = torch.cos(position_id * frequencies)

        self.register_buffer("positional_encodings", positional_encodings)

        self.dropout = nn.Dropout(p=GptConfig.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_encodings = self.positional_encodings[: x.shape[1]]
        return self.dropout(pos_encodings + x)

# ------------------------------ NanoGPT Module ------------------------------ #
class NanoGpt(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        buffer_size: int,
        n_blocks: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.buffer_size = buffer_size
        self.n_blocks = n_blocks

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_embed
        )

        if GptConfig.use_sinusoidal:
            self.positional_encodings = PositionalEncoding(
                n_embed=n_embed, max_seq_len=GptConfig.buffer_size
            )
        else:
            self.positional_encodings = nn.Embedding(
                num_embeddings=GptConfig.buffer_size, embedding_dim=n_embed
            )

        self.blocks = nn.Sequential(
            *[AttentionBlock(n_embed=n_embed, n_heads=n_heads) for _ in range(n_blocks)]
        )
        self.ln = nn.LayerNorm((n_embed,))
        self.lm_head = nn.Sequential(
            nn.Linear(n_embed, n_embed // 2), nn.GELU(), nn.Linear(n_embed // 2, vocab_size)
        )

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        B, T = input_ids.shape
        tok_emb = self.token_embeddings(input_ids) # (B, T, C)
        if GptConfig.use_sinusoidal:
            x = self.positional_encodings.forward(tok_emb) # (B, T, C) -> (B, T, C)
        else:
            x = tok_emb + self.positional_encodings(
                torch.arange(T, dtype=torch.long, device=input_ids.shape)
            ) # (B, T, C) -> (B, T, C)
        x = self.blocks(x)
        x = self.ln(x)
        x = self.lm_head(x)

        loss = None
        if labels is not None:
            B, T, C = x.shape
            loss = F.cross_entropy(x.view(B * T, C), labels.view(B * T))

        return x, loss

    def generate(
        self, ids: torch.Tensor, max_len: int, temperature: float = 0.7
    ) -> int:
        for _ in range(max_len):
            ids_cond = ids[:, -GptConfig.buffer_size :]
            logits, _ = self.forward(input_ids=ids_cond, labels=None)
            logit = logits[:, -1, :]
            probs = F.softmax(logit, dim=-1)
            val, idx = torch.topk(probs, k=int(probs.shape[1] * temperature), dim=-1)
            # print(val[0][0])
            idx_next = torch.multinomial(val, num_samples=1)
            idx_next = idx[:, idx_next][0]
            ids = torch.cat([ids, idx_next], dim=-1)
            if idx_next == 0:
                break
        return ids


if __name__ == "__main__":
    model = NanoGpt(
        vocab_size=GptConfig.vocab_size,
        n_embed=GptConfig.n_embed,
        n_heads=GptConfig.n_head,
        buffer_size=GptConfig.buffer_size,
        n_blocks=GptConfig.n_layers,
    )

    print(model)