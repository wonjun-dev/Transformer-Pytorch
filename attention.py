# attention functions
import math
import torch
import torch.nn.functional as F


def scale_dot_product_attention(Q, K, V):
    # input Q, K, V -> (N, S, E)
    # output QK^TV -> (N, S, E)
    scale = math.sqrt(K.size()[-1])
    scaled_weight = torch.matmul(Q, K.transpose(2, 1).contiguous()) / scale    # (QK^T)/scale -> (N, S, S), transpose 결과 contiguous 하지 않음.
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


    attention = scale_dot_product_attention(Q, K, V)
    print(attention.size())