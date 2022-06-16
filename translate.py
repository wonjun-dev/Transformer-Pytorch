"""Translation for test mode"""
from asyncio import transports
import torch
import math

def translation(src, max_len, model):
    # forward src sentence
    enc_in_emb = model.src_embedding(src.long()) * math.sqrt(model.d_model)
    enc_in_emb = model.pe(enc_in_emb)
    enc_out = model.encoder(enc_in_emb)
    
    dec_in = torch.LongTensor([[2]])    # start from [BOS] token
    dec_out = torch.zeros((1, max_len), dtype=torch.int16)
    for t in range(max_len):
        dec_in_emb = model.tgt_embedding(dec_in.long()) * math.sqrt(model.d_model)
        dec_in_emb = model.pe(dec_in_emb)

        _dec_out = model.decoder(dec_in_emb, enc_out)
        _dec_out = model.linear(_dec_out)
        _, topi = _dec_out[:, t].topk(k=1, dim=-1)
        topi = topi.view(-1).int()

        dec_out[:, t] = topi
        dec_in = torch.column_stack((dec_in, topi.detach().view(-1)))

        if topi == 3:   # break condition: [EOS]
            break

    dec_out = dec_out.view(-1)

    return dec_out[:t+1]


if __name__ == "__main__":
    import sentencepiece as spm
    from model import Transformer

    # load tokenizer
    src_sp = spm.SentencePieceProcessor()
    src_sp.load('./data/en.model')
    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.load('./data/de.model')

    # load trained model
    ckpt = 'ckpt/epoch=10-step=2497.ckpt'
    model = Transformer.load_from_checkpoint(ckpt)
    model.freeze()

    while True:
        src = input('Enter text to be translated in to German : ').rstrip('\n')
        token_ids = torch.tensor(list(src_sp.encode_as_ids(src))).unsqueeze(0)   # raw text to tensor
        trans = translation(token_ids, max_len=50, model=model).tolist()
        trans = tgt_sp.detokenize(trans)
        print(trans)
        

