import math
from operator import itemgetter
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchtext.data.metrics import bleu_score

from module import Encoder, Decoder
from positional_encoding import PositionalEncoding
from data_manager import ToSentencePiece
from lr_scheduler import CosineAnnealingWarmUpRestarts

class Transformer(pl.LightningModule):
    def __init__(self, max_len=200, d_model=512, vocab_size=5000, p=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(vocab_size, d_model)  # TODO src, tgt vocab 사이즈 따로
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(max_len, d_model)
        self.dropout_1 = nn.Dropout(p)
        self.dropout_2 = nn.Dropout(p)
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(d_model=d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding


    def forward(self, enc_in, dec_in):
        enc_in_emb = self.src_embedding(enc_in.long()) * math.sqrt(self.d_model)
        enc_in_emb = self.pe(enc_in_emb)
        enc_in_emb = self.dropout_1(enc_in_emb)

        enc_out = self.encoder(enc_in_emb)

        dec_in_emb = self.tgt_embedding(dec_in.long()) * math.sqrt(self.d_model)
        dec_in_emb = self.pe(dec_in_emb)
        dec_in_emb = self.dropout_2(dec_in_emb)

        dec_out = self.decoder(dec_in_emb, enc_out)    # (batch_size, tgt_max_len, d_model)
        dec_out = self.linear(dec_out)  # (batch_size, tgt_max_len, vocab_size)
        # x = F.log_softmax(x, dim=-1)
        return dec_out
        

    def training_step(self, batch, batch_idx):
        #TODO attention key padding mask
        src, tgt = batch
        tgt_out = tgt[:, 1:]

        dec_in = tgt[:, :-1]
        score = self(src, dec_in)    # (batch_size, tgt_max_len, vocab_size)
        loss = self.loss_fn(score.reshape(-1, score.shape[-1]), tgt_out.reshape(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_out = tgt[:, 1:]

        dec_in = tgt[:, :-1]
        score = self(src, dec_in)    # (batch_size, tgt_max_len, vocab_size)
        loss = self.loss_fn(score.reshape(-1, score.shape[-1]), tgt_out.reshape(-1))
        self.log('valid_loss', loss)


        # TODO BLEU score
        # tgt_out = tgt_out.tolist()
        # greedy_out = greedy_out.tolist()
        # tgt_text = self.sp.tgt_detokenize(tgt_out)
        # preds_txt = self.sp.tgt_detokenize(greedy_out)
        # self.log('val_bleu_score', bleu_score(preds_txt[0].split(), [tgt_text[0].split()]))

        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0000001)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=5, T_mult=1, eta_max=0.001, T_up=2, gamma=0.8)
        return [optimizer], [scheduler]
