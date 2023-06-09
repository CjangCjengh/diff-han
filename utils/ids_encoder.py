import torch
import torch.nn as nn


class IdsEncoder(nn.Module):
    def __init__(self, num_tokens, attention_dim, embedding_dim=64, num_heads=4):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_tokens, embedding_dim)
        self.attention_layer = nn.MultiheadAttention(embedding_dim, num_heads)
        self.linear_layer = nn.Linear(embedding_dim, attention_dim)

    def forward(self, x):
        e = self.embedding_layer(x)
        a, _ = self.attention_layer(e, e, e)
        b = self.linear_layer(a)
        v = torch.mean(b, dim=0)
        return v

