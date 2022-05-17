import math
from operator import itemgetter
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl

from module import Encoder, Decoder
from positional_encoding import PositionalEncoding

class Transformer(pl.LightningModule):
    def __init__(self, max_len, d_model, vocab_size, p=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(vocab_size, d_model)  # TODO src, tgt vocab 사이즈 따로
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(max_len, d_model).encoding()
        self.dropout_1 = nn.Dropout(p)
        self.dropout_2 = nn.Dropout(p)
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(d_model=d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        self.lr = 0.001 # TODO transformer learning rate scheduler

    def forward(self, x, tgt):
        x = self.src_embedding(x.long()) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size()[1], :]
        x = self.dropout_1(x)

        x = self.encoder(x)
        
        tgt = self.tgt_embedding(tgt.long()) * math.sqrt(self.d_model)
        tgt = tgt + self.pe[:, :tgt.size()[1], :]
        tgt = self.dropout_2(tgt)

        x = self.decoder(tgt, x)    # (batch_size, tgt_max_len, d_model)
        x = self.linear(x)  # (batch_size, tgt_max_len, vocab_size)
        # x = F.log_softmax(x, dim=-1)

        return x
        

    def training_step(self, batch, batch_idx):
        #TODO padding mask
        src, tgt = batch
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        preds = self(src, tgt_in)    # (batch_size, tgt_max_len, vocab_size)
        loss = self.loss_fn(preds.reshape(-1, preds.shape[-1]), tgt_out.reshape(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO auto regressive
        src, tgt = batch
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        preds = self(src, tgt_in)
        loss = self.loss_fn(preds.reshape(-1, preds.shape[-1]), tgt_out.reshape(-1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # TODO auto regressive
        src, tgt = batch
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        preds = self(src, tgt_in)
        loss = self.loss_fn(preds.reshape(-1, preds.shape[-1]), tgt_out.reshape(-1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer