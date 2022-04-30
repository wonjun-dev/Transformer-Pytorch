
from torchtext.datasets import Multi30k
import pytorch_lightning as pl

# dummy input
import torch

from model import Transformer

def main():
    # train_iter, valid_iter, test_iter = Multi30k()
    dummy_src = torch.rand((32, 10, 256)) # N, S, E
    dummy_tgt = torch.rand((32, 20, 256))   

    transformer = Transformer()
    out = transformer(dummy_src)
    print(out.size())
    # trainer = pl.Trainer()
    # trainer.fit(transformer)


if __name__ == '__main__':
    main()