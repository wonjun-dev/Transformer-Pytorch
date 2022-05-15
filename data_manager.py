import torch
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from typing import List

# test
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k


class ToSentencePiece():
    """
    Covert raw text batch to sentencepiece idxs batch.
    """

    def __init__(self):
        self.src_sp = spm.SentencePieceProcessor()
        self.tgt_sp = spm.SentencePieceProcessor()
        self.src_sp.load('./data/en.model')
        self.tgt_sp.load('./data/de.model')
        self.BOS_IDX = self.src_sp.bos_id()
        self.EOS_IDX = self.src_sp.eos_id()
        self.PAD_IDX = self.src_sp.pad_id()


    def _src_to_tensor(self, txt):
        token_ids = torch.tensor(list(self.src_sp.encode_as_ids(txt)))
        return torch.cat((torch.tensor([self.BOS_IDX]), token_ids, torch.tensor([self.EOS_IDX])))


    def _tgt_to_tensor(self, txt):
        token_ids = torch.tensor(self.tgt_sp.encode_as_ids(txt))
        return torch.cat((torch.tensor([self.BOS_IDX]), token_ids, torch.tensor([self.EOS_IDX])))


    def __call__(self, batch):
        src_batch, tgt_batch = [], []
        for raw_src, raw_tgt in batch:
            src_batch.append(self._src_to_tensor(raw_src.rstrip('\n')))
            tgt_batch.append(self._tgt_to_tensor(raw_tgt.rstrip('\n')))
        
        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX).transpose(1, 0).contiguous()   # padding and batch first
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX).transpose(1, 0).contiguous()

        return src_batch, tgt_batch



if __name__ == '__main__':
    train_iter = Multi30k(split='train', language_pair=('en', 'de'))
    train_dataloader = DataLoader(train_iter, batch_size=10, collate_fn=ToSentencePiece())

    for src, tgt in train_dataloader:
        print(src.size())
        print(src)
