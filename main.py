from multiprocessing import dummy
from torchtext.datasets import Multi30k
import pytorch_lightning as pl

# dummy input
import torch
from torch.utils.data import DataLoader

from model import Transformer
from data_manager import ToSentencePiece

def main():
    train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=('en', 'de'))
    train_dataloader = DataLoader(train_iter, batch_size=32, collate_fn=ToSentencePiece())
    valid_dataloader = DataLoader(valid_iter, batch_size=32, collate_fn=ToSentencePiece())
    test_dataloader = DataLoader(test_iter, batch_size=32, collate_fn=ToSentencePiece())
    dummy_src, dummy_tgt = next(iter(train_dataloader))
    # dummy_src = torch.rand((32, 10, 512)) # N, S, E
    # dummy_tgt = torch.rand((32, 10, 512))   

    transformer = Transformer(max_len=200, d_model=512, vocab_size=5000)

    # out = transformer(dummy_src, dummy_tgt)
    # print(out.size())
    pl.seed_everything(0)
    trainer = pl.Trainer()
    trainer.fit(transformer, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    main()