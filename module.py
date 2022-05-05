from torch import nn
import torch.nn.functional as F
from attention import scale_dot_product_attention
from utils import MultiInputSequential


# test
import torch

class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.wq = nn.Linear(d_model, 512, bias=False)    # 2번째 차원: d_model/n_head
        self.wk = nn.Linear(d_model, 512, bias=False)
        self.wv = nn.Linear(d_model, 512, bias=False)
        self.ff_1 = nn.Linear(512, 2048) # d_ff = 2048
        self.ff_2 = nn.Linear(2048, 512)
        self.attention = scale_dot_product_attention
        self.layer_norm_1 = nn.LayerNorm(512)  # embedding_dim 
        self.layer_norm_2 = nn.LayerNorm(512)  # embedding_dim 
    
    def forward(self, x):
        # sublayer 1 #
        identity = x # residual connection
        # attention query, key, value
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        x = self.attention(q, k, v) # TODO multi head attention
        x = x + identity # residual connection
        x = self.layer_norm_1(x)

        # sublayer 2 #
        identity = x # residual connection
        # feedforward
        x = self.ff_2(F.relu(self.ff_1(x))) # FFN(x) = max(0, xW1+b1)W2+b2
        x = x + identity
        x = self.layer_norm_2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.wq_1 = nn.Linear(d_model, 512, bias=False) # for self-attention
        self.wk_1 = nn.Linear(d_model, 512, bias=False) # for self-attention
        self.wv_1 = nn.Linear(d_model, 512, bias=False) # for self-attention
        self.wq_2 = nn.Linear(d_model, 512, bias=False) # for encoder-decoder attention
        self.wk_2 = nn.Linear(d_model, 512, bias=False) # for encoder-decoder attention
        self.wv_2 = nn.Linear(d_model, 512, bias=False) # for encoder-decoder attention
        self.ff_1 = nn.Linear(512, 2048) # d_ff = 2048
        self.ff_2 = nn.Linear(2048, 512)
        self.attention = scale_dot_product_attention
        self.layer_norm_1 = nn.LayerNorm(512)
        self.layer_norm_2 = nn.LayerNorm(512)
        self.layer_norm_3 = nn.LayerNorm(512)
    
    def forward(self, x, enc_out):
        # sublayer 1 #
        identity = x # residual connection
        # attention query, key, value
        q = self.wq_1(x)
        k = self.wk_1(x)
        v = self.wv_1(x)

        x = self.attention(q, k, v, masking=True)   # masked attention
        x = x + identity
        x = self.layer_norm_1(x)

        # sublayer 2 #
        identity = x
        q = self.wq_2(x)
        enc_k = self.wk_2(enc_out)
        enc_v = self.wv_2(enc_out)
        x = self.attention(q, enc_k, enc_v) # encoder-decoder attention
        x = x + identity
        x = self.layer_norm_2(x)

        # sublayer 3 #
        identity = x
        x = self.ff_2(F.relu(self.ff_1(x))) # FFN(x) = max(0, xW1+b1)W2+b2
        x = x + identity
        x = self.layer_norm_3(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_stack=6):
        super().__init__()
        self.layer = nn.Sequential()
        for i in range(n_stack):
            self.layer.add_module(f'EncoderLayer_{i}', EncoderLayer(d_model=d_model))

    def forward(self, x):
        x = self.layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_stack=6):
        super().__init__()
        self.layer = MultiInputSequential()
        for i in range(n_stack):
            self.layer.add_module(f'DecoderLayer_{i}', DecoderLayer(d_model=d_model))

    def forward(self, x, enc_out):
        x = self.layer(x, enc_out)
        return x


if __name__ == '__main__':
    input = torch.rand(2, 5, 512)
    # el = EncoderLayer(512)
    # out = el(input)
    # print(out)
    # print(out.size())

    dl = DecoderLayer(512)
    out = dl(input, input)
    print(out)
    print(out.size())