from torch import nn

class MultiInputSequential(nn.Sequential):
    """
    Encoder-Decoder Attention 을 하기 위해서 DecoderLayer forward가 Encoder의 아웃풋을 받기 위한 클래스.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        for module in self:
            input = module(x, *args, **kwargs)
        return input
