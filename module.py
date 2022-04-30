from torch import nn
from attention import scale_dot_product_attention
from positional_encoding import sin_encoding

# test
import torch

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(256, 64, bias=False)
        self.wk = nn.Linear(256, 64, bias=False)
        self.wv = nn.Linear(256, 64, bias=False)
        self.attention = scale_dot_product_attention
        self.feedforward = None
    
    def forward(self, x):
        q = self.wq(x)
        k = self.wq(x)
        v = self.wq(x)
        
        x = self.attention(q, k, v)
        # x = self.feedforward(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass


class Encoder(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.pos_encoding = sin_encoding
        self.layer = nn.Sequential()
        for i in range(n):
            self.layer.add_module(f'EncoderLayer_{i}', EncoderLayer())

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


if __name__ == '__main__':
    input = torch.rand(2, 5, 256)
    el = EncoderLayer()
    out = el(input)
    print(out)
    print(out.size())