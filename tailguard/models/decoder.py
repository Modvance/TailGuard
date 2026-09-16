"""Transformer decoder blocks required by the TailGuard reconstruction core."""

import torch
import torch.nn as nn


class BottleneckMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, inputs):
        outputs = self.drop(inputs)
        outputs = self.fc1(outputs)
        outputs = self.act(outputs)
        outputs = self.drop(outputs)
        outputs = self.fc2(outputs)
        return self.drop(outputs)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.act(outputs)
        outputs = self.drop(outputs)
        outputs = self.fc2(outputs)
        return self.drop(outputs)


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        del qk_scale
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, inputs, attn_mask=None):
        del attn_mask
        batch_size, sequence_length, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            channels // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = torch.nn.functional.elu(query) + 1.0
        key = torch.nn.functional.elu(key) + 1.0
        key_value = torch.einsum("...sd,...se->...de", key, value)
        normalizer = 1.0 / torch.einsum("...sd,...d->...s", query, key.sum(dim=-2))
        outputs = torch.einsum("...de,...sd,...s->...se", key_value, query, normalizer)
        outputs = outputs.transpose(1, 2).reshape(batch_size, sequence_length, channels)
        outputs = self.proj(outputs)
        return self.proj_drop(outputs), key_value


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn=LinearAttention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop,
        )

    def forward(self, inputs, return_attention=False, attn_mask=None):
        attention_output, attention = self.attn(self.norm1(inputs), attn_mask=attn_mask)
        outputs = inputs + self.drop_path(attention_output)
        outputs = outputs + self.drop_path(self.mlp(self.norm2(outputs)))
        if return_attention:
            return outputs, attention
        return outputs


# Backward-compatible aliases used by the original TailGuard configuration.
bMlp = BottleneckMLP
LinearAttention2 = LinearAttention

__all__ = ["Block", "BottleneckMLP", "LinearAttention", "LinearAttention2", "bMlp"]
