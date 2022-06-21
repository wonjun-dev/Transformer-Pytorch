from multiprocessing import dummy
from torchtext.datasets import Multi30k
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# dummy input
import torch
from torch.utils.data import DataLoader

from model import Transformer
from data_manager import ToSentencePiece

def main():
    device = 'cuda:0'
    train_iter, valid_iter, test_iter = Multi30k(root='./data/', split=('train', 'valid', 'test'), language_pair=('en', 'de'))
    train_dataloader = DataLoader(train_iter, batch_size=128, collate_fn=ToSentencePiece(), shuffle=True)
    valid_dataloader = DataLoader(valid_iter, batch_size=128, collate_fn=ToSentencePiece())
    test_dataloader = DataLoader(test_iter, batch_size=10, collate_fn=ToSentencePiece())

    # dummy_src = torch.rand((32, 10, 512)) # N, S, E
    # dummy_tgt = torch.rand((32, 10, 512))   
    
    transformer = Transformer(max_len=200, d_model=512, vocab_size=5000)
    transformer = transformer.to(device)

    # out = transformer(dummy_src, dummy_tgt)
    # print(out.size())
    pl.seed_everything(0)
    checkpoint_callback = ModelCheckpoint(dirpath='ckpt/', save_top_k=2, monitor='valid_loss')
    trainer = pl.Trainer(devices=[0], accelerator='gpu', max_epochs=20, callbacks=[checkpoint_callback])
    trainer.fit(transformer, train_dataloader, valid_dataloader)



if __name__ == '__main__':
    main()