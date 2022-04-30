import pytorch_lightning
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from module import Encoder, Decoder

class Transformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
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