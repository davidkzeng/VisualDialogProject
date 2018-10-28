import torch
from torch import nn
from torch.nn import functional as F

from utils import DynamicRNN


class HierarchicalRecurrentEncoder(nn.Module):

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Encoder specific arguments')
        parser.add_argument('-img_feature_size', default=4096,
                                help='Channel size of image feature')
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

        self.word_embed = nn.Embedding(args.vocab_size, args.embed_size, padding_idx=0)
        ques_img_feature_size = args.embed_size + args.img_feature_size
        self.ques_img_rnn = nn.LSTM(ques_img_feature_size, args.rnn_hidden_size + args.img_feature_size,
                                args.num_layers,
                                batch_first=True, dropout=args.dropout)
        self.ques_img_rnn = DynamicRNN(self.ques_img_rnn)

        self.hist_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size, args.num_layers,
                        batch_first=True, dropout=args.dropout)
        self.hist_rnn = DynamicRNN(self.hist_rnn)

        fusion_size = args.img_feature_size + args.rnn_hidden_size * 2
        self.fusion = nn.Linear(fusion_size, args.rnn_hidden_size)

        """
        if args.weight_init == 'xavier':
            nn.init.xavier_uniform(self.fusion.weight.data)
        elif args.weight_init == 'kaiming':
            nn.init.kaiming_uniform(self.fusion.weight.data)
        nn.init.constant(self.fusion.bias.data, 0)
        """
        self.dialog_rnn = nn.LSTM(args.rnn_hidden_size, args.rnn_hidden_size, args.num_layers,
                        batch_first=True, dropout=args.dropout)



    def forward(self, batch):
        img = batch['img_feat']
        ques = batch['ques']
        hist = batch['hist']
        ques_len = batch['ques_len']
        hist_len = batch['hist_len']

        batch_size = ques.size(0)
        round_size = ques.size(1)
        ques_size = ques.size(2)

        ques_embed = self.word_embed(ques)
        # repeat image feature vectors to be provided for every round
        img = img.view(batch_size, 1, 1, -1)
        img = img.repeat(1, round_size, ques_size, 1)
        print(img.size(), ques_embed.size(), hist.size(), ques_len.size())

        ques_img = torch.cat((img, ques_embed), 3)
        print(ques_img.size())
