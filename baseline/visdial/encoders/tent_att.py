import torch
from torch import nn
from torch.nn import functional as F

from utils import DynamicRNN


class TentativeAttentionEncoder(nn.Module):

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Encoder specific arguments')
        parser.add_argument('-img_embed_size', default=512,
                                help='Embed size for image')
        parser.add_argument('-embed_size', default=300,
                                help='Size of the input word embedding')
        parser.add_argument('-rnn_hidden_size', default=512,
                                help='Size of the multimodal embedding')
        parser.add_argument('-num_layers', default=2,
                                help='Number of layers in LSTM')
        parser.add_argument('-max_history_len', default=60,
                                help='Size of the multimodal embedding')
        parser.add_argument('-dropout', default=0.5, help='Dropout')
        return parser

    def __init__(self, args):
        super().__init__()
        self.args = args
        img_num_features = 512
        img_size = 49
        self.args.img_feature_size = (img_size, img_num_features)

        self.word_embed = nn.Embedding(args.vocab_size, args.embed_size, padding_idx=0)

        self.ques_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size, args.num_layers,
                                batch_first=True, dropout=args.dropout)
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        self.hist_q_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size, args.num_layers,
                                batch_first=True, dropout=args.dropout)
        self.hist_q_rnn = DynamicRNN(self.hist_q_rnn)
        self.hist_a_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size, args.num_layers,
                                batch_first=True, dropout=args.dropout)

        self.qa_fusion = nn.Linear(args.rnn_hidden_size * 2, args.rnn_hidden_size)
        self.hist_rnn = nn.LSTM(args.rnn_hidden_size, args.rnn_hidden_size, args.num_layers,
                        batch_first=True, dropout=args.dropout)

        self.qh_fusion = nn.Linear(args.rnn_hidden_size * 2, args.rnn_hidden_size)

        self.c_projection = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size)
        self.f_projection = nn.Linear(img_num_features, args.rnn_hidden_size)

        # Transpose + Matrix Multiplication Stuff in Forward
        # Softmax stuff in forward (nn.softmax)
        # Elementwise multiplication to get attended image

        # Currently f^att_t, alpha_t, c_t
        self.final_concat_size = img_num_features + img_size + args.rnn_hidden_size
        self.final_fusion = nn.Linear(final_concat_size, args.rnn_hidden_size)



    def forward(self, batch):
        img = batch['img_feat']
        ques = batch['ques']
        hist = batch['hist']
        ques_len = batch['ques_len']
        hist_len = batch['hist_len']

        batch_size = ques.size(0)
        # Typically 10 questions per round
        round_size = ques.size(1)
        # Max length of questions
        ques_size = ques.size(2)

        ques = ques.view(-1, ques.size(2))
        ques_embed = self.word_embed(ques)
        ques_embed = self.ques_rnn(ques_word_embed)
        # embed history
        
        # Flatten batch_size * round
        hist = hist.view(-1, hist.size(2))
        hist_embed = self.word_embed(hist)
        hist_embed = self.hist_q_rnn(hist_embed, batch['hist_len'])
        hist_embed = hist_embed.view(-1, round_size, hist_embed.size(1))
        hist_embed = hist_embed.permute(1, 0, 2)
        hist_hier, _ = self.hist_rnn(hist_embed, None)
        hist_hier = hist_hier.permute(1, 0, 2)
        hist_hier = hist_hier.view(-1, hist_hier.size(2))

        q_hist_fused = torch.cat((ques_embed, hist_embed), 1)
        q_hist_fused = self.qh_fusion(q_hist_fused)
        
        c_proj = self.c_projection(q_hist_fused)


        return output
