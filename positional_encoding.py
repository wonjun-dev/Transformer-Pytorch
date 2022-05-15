# positional encoding class
import torch

class PositionalEncoding():
    def __init__(self, max_len, d_model):
        self.max_len = max_len
        self.d_model = d_model
        self.pe = self.encoding()

    def encoding(self):
        """
        Return
            pe: (torch.tensor) |  (1, max_len, d_model)
        """
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(self.max_len).unsqueeze(1)
        exponent = torch.arange(0, self.d_model, 2) / self.d_model
        div = torch.pow(10000*torch.ones_like(exponent), exponent)
        pe[:, ::2] = torch.sin(position / div)
        pe[:, 1::2] = torch.cos(position / div)
        pe.unsqueeze_(0)

        # TODO dropout

        return pe


if __name__ == '__main__':
    p = PositionalEncoding(10, 128)
    print(p.pe.size())
    print(p.pe)