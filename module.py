from typing import MutableMapping
from torch import nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from attention import EncoderDecoderAttention
from utils import MultiInputSequential


# test
import torch

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attention_layer = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.ff_1 = nn.Linear(d_model, 2048) # d_ff = 2048
        self.ff_2 = nn.Linear(2048, d_model)
        self.layer_norm_1 = nn.LayerNorm(d_model)  # embedding_dim 
        self.layer_norm_2 = nn.LayerNorm(d_model)  # embedding_dim 
        # TODO dropout
    
    def forward(self, x):
        # sublayer 1 #
        identity = x
        x = self.attention_layer(x)
        x = x + identity # residual connection
        x = self.layer_norm_1(x)

        # sublayer 2 #
        identity = x
        # feedforward
        x = self.ff_2(F.relu(self.ff_1(x))) # FFN(x) = max(0, xW1+b1)W2+b2
        x = x + identity
        x = self.layer_norm_2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.masked_attention_layer = MultiHeadAttention(d_model=d_model, n_head=n_head, masking=True)
        self.ende_attention_layer = EncoderDecoderAttention(d_model=d_model, n_head=n_head)
        self.ff_1 = nn.Linear(512, 2048) # d_ff = 2048
        self.ff_2 = nn.Linear(2048, 512)
        self.layer_norm_1 = nn.LayerNorm(512)
        self.layer_norm_2 = nn.LayerNorm(512)
        self.layer_norm_3 = nn.LayerNorm(512)
        # TODO dropout
    
    def forward(self, x, enc_out):
        # sublayer 1 #
        identity = x # residual connection
        x = self.masked_attention_layer(x)
        x = x + identity
        x = self.layer_norm_1(x)

        # sublayer 2 #  # TODO encoder-decoder mutli head attention
        identity = x
        x = self.ende_attention_layer(x, enc_out)
        x = x + identity
        x = self.layer_norm_2(x)

        # sublayer 3 #
        identity = x
        x = self.ff_2(F.relu(self.ff_1(x))) # FFN(x) = max(0, xW1+b1)W2+b2
        x = x + identity
        x = self.layer_norm_3(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_stack=6, n_head=8):
        super().__init__()
        self.layer = nn.Sequential()
        for i in range(n_stack):
            self.layer.add_module(f'EncoderLayer_{i}', EncoderLayer(d_model=d_model, n_head=n_head))

    def forward(self, x):
        x = self.layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_stack=6, n_head=8):
        super().__init__()
        self.layer = MultiInputSequential()
        for i in range(n_stack):
            self.layer.add_module(f'DecoderLayer_{i}', DecoderLayer(d_model=d_model, n_head=n_head))

    def forward(self, x, enc_out):
        x = self.layer(x, enc_out)
        return x


if __name__ == '__main__':
    input = torch.rand(2, 5, 512)
    el = EncoderLayer(512, 8)
    out = el(input)
    print(out)
    print(out.size())

    en = Encoder(512, 6, 8)
    out = en(input)
    print(out)
    print(out.size())

    dl = DecoderLayer(512, 8)
    out = dl(input, input)
    print(out)
    print(out.size())

    de = Decoder(512, 6, 8)
    out = de(input, input)
    print(out)
    print(out.size())