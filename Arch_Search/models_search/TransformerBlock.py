import torch.nn as nn
import numpy as np
import torch

class matmul(nn.Module):
    def __init__(self):
        super(matmul, self).__init__()

    def forward(self, x1, x2):
        x = np.dot(x1, x2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.mat = matmul()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return x


class TransEncoderBlock(nn.Module):

    def __init__(self, dim, num_heads=1, dropout=0.1):
        super(TransEncoderBlock, self).__init__()

        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

        self.linear = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.dropout1(self.attn(x)))
        x = self.norm2(x + self.dropout2(self.linear(x)))
        return x