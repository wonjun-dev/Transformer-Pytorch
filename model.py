import pytorch_lightning
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from module import Encoder, Decoder
from positional_encoding import PositionalEncoding

class Transformer(pl.LightningModule):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = PositionalEncoding(max_len, d_model).encoding()
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(d_model=d_model)

    def forward(self, x):
        x = x + self.pe
        x = self.encoder(x)
        # x = self.decoder(x)
        return x
        

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass