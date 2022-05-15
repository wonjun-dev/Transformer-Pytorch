import math
from torch import nn
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
        self.dropout = nn.Dropout(p)
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(d_model=d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, tgt):
        x = self.src_embedding(x.long()) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size()[1], :]
        x = self.dropout(x)
        x = self.encoder(x)
        
        tgt = self.tgt_embedding(tgt.long()) * math.sqrt(self.d_model)
        tgt = tgt + self.pe[:, :x.size()[1], :]
        tgt = self.dropout(tgt)
        x = self.decoder(tgt, x)
        x = self.linear(x)
        x = F.softmax(x, dim=-1)

        return x
        

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass