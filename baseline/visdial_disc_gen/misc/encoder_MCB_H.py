import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F

class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout, emb_size):
        super(_netE, self).__init__()

        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp

        self.inp_linear = nn.Linear(emb_size, nhid)
        self.his_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        self.Wh_1 = nn.Linear(self.nhid, self.nhid)
        self.Wi_2 = nn.Linear(self.nhid, self.nhid)
        self.Wa_1 = nn.Linear(self.nhid, 1)

        self.fc1 = nn.Linear(self.nhid, self.ninp)

    def forward(self, inp_emb, his_emb, his_hidden, rnd):
        inp_emb = F.tanh(self.inp_linear(inp_emb))

        his_feat, his_hidden = self.his_rnn(his_emb, his_hidden)
        his_feat = his_feat[-1]

        his_emb_1 = self.Wh_1(his_feat).view(-1, rnd, self.nhid)

        inp_emb_2 = self.Wi_2(inp_emb).view(-1, 1, self.nhid)

        atten_emb = F.tanh(inp_emb_2.expand_as(his_emb_1) + his_emb_1)
        inp_atten_weight = F.softmax(self.Wa_1(F.dropout(atten_emb, self.d,
            training=self.training).view(-1, self.nhid)).view(-1, rnd))

        inp_attn_feat = torch.bmm(inp_atten_weight.view(-1, 1, rnd),
                                        his_feat.view(-1, rnd, self.nhid))
        inp_attn_feat = inp_attn_feat.view(-1,self.nhid)

        encoder_feat = F.tanh(self.fc1(F.dropout(inp_attn_feat, self.d,
            training = self.training)))

        return encoder_feat

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
