from multiprocessing import dummy
from torchtext.datasets import Multi30k
import pytorch_lightning as pl

# dummy input
import torch
from torch.utils.data import DataLoader

from model import Transformer
from data_manager import ToSentencePiece

def main():
    train_iter = Multi30k(split='train', language_pair=('en', 'de'))
    train_dataloader = DataLoader(train_iter, batch_size=10, collate_fn=ToSentencePiece())
    dummy_src, dummy_tgt = next(iter(train_dataloader))
    # dummy_src = torch.rand((32, 10, 512)) # N, S, E
    # dummy_tgt = torch.rand((32, 10, 512))   

    transformer = Transformer(max_len=200, d_model=512, vocab_size=5000)

    out = transformer(dummy_src, dummy_tgt)
    print(out.size())
    # trainer = pl.Trainer()
    # trainer.fit(transformer)


if __name__ == '__main__':
    main()