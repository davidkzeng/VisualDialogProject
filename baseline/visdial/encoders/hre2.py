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

        self.img_embed = nn.Linear(args.img_feature_size, args.img_embed_size)
        self.word_embed = nn.Embedding(args.vocab_size, args.embed_size, padding_idx=0)

        ques_img_feature_size = args.embed_size + args.img_embed_size
        self.ques_img_rnn = nn.LSTM(ques_img_feature_size, args.rnn_hidden_size,
                                args.num_layers,
                                batch_first=True, dropout=args.dropout)
        self.ques_img_rnn = DynamicRNN(self.ques_img_rnn)

        self.hist_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size, args.num_layers,
                        batch_first=True, dropout=args.dropout)
        self.hist_rnn = DynamicRNN(self.hist_rnn)

        # Original Paper did not use fusion
        # fusion_size = args.rnn_hidden_size * 2
        # self.fusion = nn.Linear(fusion_size, args.rnn_hidden_size)

        """
        if args.weight_init == 'xavier':
            nn.init.xavier_uniform(self.fusion.weight.data)
        elif args.weight_init == 'kaiming':
            nn.init.kaiming_uniform(self.fusion.weight.data)
        nn.init.constant(self.fusion.bias.data, 0)
        """
        self.dialog_rnn = nn.LSTM(args.rnn_hidden_size * 2, args.rnn_hidden_size, args.num_layers,
                        batch_first=True, dropout=args.dropout)
        # self.dropout = nn.Dropout(p=args.dropout)



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

        ques_embed = self.word_embed(ques)
        img = self.img_embed(img)

        # repeat image feature vectors to be provided for every round
        img = img.view(batch_size, 1, 1, -1)
        img = img.repeat(1, round_size, ques_size, 1)

        ques_img = torch.cat((img, ques_embed), 3)
        # Flatten to batch_size * round
        ques_img = ques_img.view(-1, ques_size, ques_img.size(3))

        ques_img_encode = self.ques_img_rnn(ques_img, ques_len)

        # embed history
        # Flatten batch_size * round
        hist = hist.view(-1, hist.size(2))
        hist_embed = self.word_embed(hist)
        hist_embed = self.hist_rnn(hist_embed, batch['hist_len'])

        fused_vector = torch.cat((ques_img_encode, hist_embed), 1)
        # Dropout not used in the torch implementation of HRE
        # fused_vector = self.dropout(fused_vector)

        # Copied the tanh over from lf.py but unclear if needed since this is an intermediate layer
        # fused_embedding = F.tanh(self.fusion(fused_vector))

        # fused_embedding = self.fusion(fused_vector)
        # fused_embedding = fused_embedding.view(-1, round_size, fused_embedding.size(1))

        # LSTM expects dimensions to be (seq_len, batch, input_size)
        # Expand to batch_size * round * embed_size
        fused_vector = fused_vector.view(-1, round_size, fused_vector.size(1))
        fused_vector = fused_vector.permute(1, 0, 2)
        output, final = self.dialog_rnn(fused_vector, None)

        # LSTM outputs tensor with dimensions (seq_len, batch, rnn_hidden_size)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(-1, output.size(2))

        # Unclear if we want this tanh
        # output = torch.tanh(output)

        return output
