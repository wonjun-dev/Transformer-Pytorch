# attention functions
import math
import torch
import torch.nn.functional as F


def scale_dot_product_attention(Q, K, V, masking=False):
    # input Q, K, V -> (N, S, E)
    # output QK^TV -> (N, S, E)
    scale = math.sqrt(K.size()[-1])
    scaled_weight = torch.matmul(Q, K.transpose(2, 1).contiguous()) / scale    # (QK^T)/scale -> (N, S, S), transpose 결과 contiguous 하지 않음.

    if masking:
        n = scaled_weight.size()[-1]
        mask = torch.ones(n, n)
        mask = torch.tril(mask) # lower triangle
        scaled_weight = scaled_weight.masked_fill_(mask==0, float('-inf'))

    softmax= F.softmax(scaled_weight, dim=-1)
    attention = torch.matmul(softmax, V) 
    return attention
    

def multi_head_attention():
    pass

def encoder_decoder_attention():
    pass


if __name__ == '__main__':
    Q = torch.randn(2, 5, 10)
    K = torch.randn(2, 5, 10)
    V = torch.randn(2, 5, 10)


    attention1 = scale_dot_product_attention(Q, K, V)
    attention2 = scale_dot_product_attention(Q, K, V, masking=True)
    print(attention1)
    print(attention2)