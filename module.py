from torch import nn
from attention import scale_dot_product_attention


# test
import torch

class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.wq = nn.Linear(d_model, 512, bias=False)    # 2번째 차원: d_model/n_head
        self.wk = nn.Linear(d_model, 512, bias=False)
        self.wv = nn.Linear(d_model, 512, bias=False)
        self.attention = scale_dot_product_attention
        self.feedforward = None
        self.layer_norm = nn.LayerNorm(512)  # embedding_dim 
    
    def forward(self, x):
        # residual connection
        identity = x

        # attention query, key, value
        q = self.wq(x)
        k = self.wq(x)
        v = self.wq(x)
        
        x = self.attention(q, k, v) # TODO multi head attention
        x = x + identity # residual connection
        x = self.layer_norm(x)
        # x = self.feedforward(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass


class Encoder(nn.Module):
    def __init__(self, d_model, n_stack=1):   # TODO N stack 
        super().__init__()
        self.layer = nn.Sequential()
        for i in range(n_stack):
            self.layer.add_module(f'EncoderLayer_{i}', EncoderLayer(d_model=d_model))

    def forward(self, x):
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