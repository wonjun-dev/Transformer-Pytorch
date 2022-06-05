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

        # Data
        self.sp = ToSentencePiece()
        self.VOCAB_SIZE = vocab_size
        self.MAX_LEN = 50

    def forward(self, enc_in, dec_in, is_train=True):
        enc_in_emb = self.src_embedding(enc_in.long()) * math.sqrt(self.d_model)
        enc_in_emb = enc_in_emb + self.pe[:, :enc_in_emb.size()[1], :]
        enc_in_emb = self.dropout_1(enc_in_emb)

        enc_out = self.encoder(enc_in_emb)

        if is_train:    # Teacher forcing
            dec_in_emb = self.tgt_embedding(dec_in.long()) * math.sqrt(self.d_model)
            dec_in_emb = dec_in_emb + self.pe[:, :dec_in_emb.size()[1], :]
            dec_in_emb = self.dropout_2(dec_in_emb)

            dec_out = self.decoder(dec_in_emb, enc_out)    # (batch_size, tgt_max_len, d_model)
            dec_out = self.linear(dec_out)  # (batch_size, tgt_max_len, vocab_size)
            # x = F.log_softmax(x, dim=-1)
            return dec_out

        else:   # Auto regressive (greedy docoding)
            batch_size = dec_in.size()[0]
        
            dec_out_v = torch.zeros((batch_size, self.MAX_LEN, self.VOCAB_SIZE))  # unormalized score
            dec_out_i = torch.zeros((batch_size, self.MAX_LEN), dtype=torch.int16)  # greedy out

            for t in range(self.MAX_LEN):
                dec_in_emb = self.tgt_embedding(dec_in.long()) * math.sqrt(self.d_model)
                _dec_out = self.decoder(dec_in_emb, enc_out)
                _dec_out = self.linear(_dec_out)
                topv, topi = _dec_out[:, t].topk(k=1, dim=-1)
                dec_out_v[:, t] = _dec_out[:, t]
                dec_out_i[:, t] = topi.view(-1).int()
                dec_in = torch.column_stack((dec_in, topi.detach().view(-1)))
        
            return dec_out_v, dec_out_i

        

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

        batch_size = src.size()[0]
        self.MAX_LEN = tgt_out.size()[1]
        dec_in = torch.LongTensor([[2] for _ in range(batch_size)])
        score, greedy_out = self(src, dec_in, is_train=False)
        
        loss = self.loss_fn(score.reshape(-1, score.shape[-1]), tgt_out.reshape(-1))
        self.log('valid_loss', loss)

        # TODO BLEU score
        tgt_out = tgt_out.tolist()
        greedy_out = greedy_out.tolist()
        # tgt_text = self.sp.tgt_detokenize(tgt_out)
        # preds_txt = self.sp.tgt_detokenize(greedy_out)
        # self.log('val_bleu_score', bleu_score(preds_txt[0].split(), [tgt_text[0].split()]))

        return loss

    # def test_step(self, batch, batch_idx):
    #     src, tgt = batch
    #     tgt_in = tgt[:, :-1]
    #     tgt_out = tgt[:, 1:]
    #     preds = self(src, tgt_in)
    #     loss = self.loss_fn(preds.reshape(-1, preds.shape[-1]), tgt_out.reshape(-1))
    #     self.log('test_loss', loss)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
