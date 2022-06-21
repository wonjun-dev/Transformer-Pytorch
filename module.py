from typing import MutableMapping
from torch import nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from attention import EncoderDecoderAttention
from utils import MultiInputSequential


# test
import torch

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, p=0.1):
        super().__init__()
        self.attention_layer = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.ff_1 = nn.Linear(d_model, 2048) # d_ff = 2048
        self.ff_2 = nn.Linear(2048, d_model)
        self.layer_norm_1 = nn.LayerNorm(d_model)  # embedding_dim 
        self.layer_norm_2 = nn.LayerNorm(d_model)  # embedding_dim 
        self.dropout_1 = nn.Dropout(p)
        self.dropout_2 = nn.Dropout(p)
        self.dropout_3 = nn.Dropout(p)
    
    def forward(self, x):
        # sublayer 1 #
        identity = x
        out_1 = self.dropout_1(self.attention_layer(x))
        out_1 = out_1 + identity # residual connection
        out_1 = self.layer_norm_1(out_1)

        # sublayer 2 #
        identity = out_1
        # feedforward
        out_2 = self.dropout_3(self.ff_2(self.dropout_2(F.relu(self.ff_1(out_1))))) # FFN(x) = max(0, xW1+b1)W2+b2
        out_2 = out_2 + identity
        out_2 = self.layer_norm_2(out_2)

        return out_2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, p=0.1):
        super().__init__()
        self.masked_attention_layer = MultiHeadAttention(d_model=d_model, n_head=n_head, masking=True)
        self.ende_attention_layer = EncoderDecoderAttention(d_model=d_model, n_head=n_head)
        self.ff_1 = nn.Linear(512, 2048) # d_ff = 2048
        self.ff_2 = nn.Linear(2048, 512)
        self.layer_norm_1 = nn.LayerNorm(512)
        self.layer_norm_2 = nn.LayerNorm(512)
        self.layer_norm_3 = nn.LayerNorm(512)
        self.dropout_1 = nn.Dropout(p)
        self.dropout_2 = nn.Dropout(p)
        self.dropout_3 = nn.Dropout(p)
        self.dropout_4 = nn.Dropout(p)
    
    def forward(self, x, enc_out):
        # sublayer 1 #
        identity = x # residual connection
        out_1 = self.dropout_1(self.masked_attention_layer(x))
        out_1 = out_1 + identity
        out_1 = self.layer_norm_1(out_1)


        # sublayer 2 #  # TODO encoder-decoder mutli head attention
        identity = out_1
        out_2 = self.dropout_2(self.ende_attention_layer(out_1, enc_out))
        out_2 = out_2 + identity
        out_2 = self.layer_norm_2(out_2)

        # sublayer 3 #
        identity = out_2
        out_3 = self.dropout_4(self.ff_2(self.dropout_3(F.relu(self.ff_1(out_2))))) # FFN(x) = max(0, xW1+b1)W2+b2
        out_3 = out_3 + identity
        out_3 = self.layer_norm_3(out_3)

        return out_3


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