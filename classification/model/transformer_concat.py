import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import *


class TransformerClf_concat(nn.Module):
    def __init__(self, ntokens, ncls, emsizes, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.ntokens = ntokens
        self.emsizes = emsizes
        self.emsize = sum(emsizes)
        self.nhead = nhead
        self.nhid = nhid
        self.dropout = dropout
        self.nlayers = nlayers
        encoder_layers = nn.TransformerEncoderLayer(self.emsize, nhead, nhid, dropout)
        self.transformer_block = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.encoders = [Encoder(ntokens[i], emsizes[i], dropout) for i in range(len(ntokens))]
        self.encoders = nn.ModuleList(self.encoders)
        self.decoder = Decoder(ncls, self.emsize, dropout)
    def forward(self, inps, labels = None):
        # (batch, seq_len)
        src = [encoder(inp) for encoder, inp in zip(self.encoders, inps)]
        src = torch.cat(src, dim=-1)
        src = self.transformer_block(src)
        # print(src.shape)
        src = src[0]
        # (batch, seq_len, ntoken)
        logits = self.decoder(src)
        prob = F.softmax(logits, dim=1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
def unit_test():
    ntokens = (10,5)
    ninp = 12
    nhid = 12
    nhead = 4
    nlayers = 2
    emsize = (12,16)
    dropout = 0.5
    ncls = 4
    model = TransformerClf_concat(ntokens, ncls, emsize, nhead, nhid, nlayers, dropout)
    model = model.cuda()
    # print(model)
    src = [torch.randint(ntokens[i], (8, 25)).cuda() for i in range(len(emsize))]
    print([src_.shape for src_ in src])
    tgt = model(src)
    print(tgt.shape)

    label = torch.randint(ncls, (8,)).cuda()
    loss,prob = model(src, label)
    print(loss)
    print(prob.shape)
    y_preds = prob.argmax(dim=1)
    print(y_preds.shape)
    print(y_preds)


if __name__ == '__main__':
    unit_test()
    pass

