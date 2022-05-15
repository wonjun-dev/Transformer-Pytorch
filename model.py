from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from module import Encoder, Decoder
from positional_encoding import PositionalEncoding

class Transformer(pl.LightningModule):
    def __init__(self, max_len, d_model, vocab_size):
        super().__init__()
        # TODO Embedding layer
        self.pe = PositionalEncoding(max_len, d_model).encoding()
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(d_model=d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, tgt):
        x = x + self.pe
        x = self.encoder(x)
        tgt = tgt + self.pe
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