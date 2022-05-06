# attention class
import math
import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, masking=False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.masking = masking
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)

        assert self.d_model % self.n_head == 0  # d_k = d_model / n_head 이므로 나누어 떨어져야 함
    

    def scale_dot_product_attention(self, Q, K, V):
        # input Q, K, V -> (batch_size, n_head, max_len, d_k)
        # QK^T -> (batch_size, n_head, max_len, max_len)
        # output QK^TV -> (batch_size, n_head, max_len, d_k)
        scale = math.sqrt(K.size()[-1])
        scaled_weight = torch.matmul(Q, K.transpose(3, 2).contiguous()) / scale # transpose 결과 contiguous 하지 않음.
        
        if self.masking:
            max_len = scaled_weight.size()[-1]
            mask = torch.ones(1, max_len, max_len)  # broadcasting mask to all heads
            mask = torch.tril(mask) # lower triangle
            scaled_weight = scaled_weight.masked_fill_(mask==0, float('-inf'))

        score= F.softmax(scaled_weight, dim=-1)
        attention = torch.matmul(score, V)

        return attention


    def forward(self, x):
        q = self.wq(x)  # (batch_size, max_len, d_model)
        k = self.wk(x)  # (batch_size, max_len, d_model)
        v = self.wv(x)  # (batch_size, max_len, d_model)

        # transform tensor # (batch_size, max_len, d_model) -> # (batch_size, n_head, max_len, d_k)
        batch_size = q.size()[0]
        max_len = q.size()[1]
        d_k = self.d_model // self.n_head
        q = q.view(batch_size, max_len, self.n_head, d_k).transpose(2, 1).contiguous()
        k = k.view(batch_size, max_len, self.n_head, d_k).transpose(2, 1).contiguous()
        v = v.view(batch_size, max_len, self.n_head, d_k).transpose(2, 1).contiguous()

        attention = self.scale_dot_product_attention(q, k, v)   # (batch_size, n_head, max_len, d_k)
        attention = attention.transpose(2, 1).contiguous()
        attention = attention.view(batch_size, max_len, -1) # Concat(head1, head2, ...)
        out = self.linear(attention)

        return out


if __name__ == '__main__':
    dummy_src = torch.rand((2, 5, 8)) # (batch_size, max_len, d_model)

    attention_layer = MultiHeadAttention(d_model=8, n_head=4)
    out1 = attention_layer(dummy_src)
    print(out1)
    masked_attention_layer = MultiHeadAttention(d_model=8, n_head=4, masking=True)
    out2 = masked_attention_layer(dummy_src)
    print(out2)