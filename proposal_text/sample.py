# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--load', type=str, default='model.pt',
                    help='path to load the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    words = [""] * 70
    for i in range(70):
        words[i] = corpus.dictionary.idx2word[data[i]]
    # print("batchify words",words)
    # print("batchify data",data[:100])
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    # print("batchify data size", data.size())
    # print("batchify data reshaped", data[:3])
    words = [[""] * data.size(1)] * 3
    for i in range(3):
        for j in range(data.size(1)):
            words[i][j] = corpus.dictionary.idx2word[data[i][j]]
    # print("batchify words reshaped",words)
    return data.to(device)

eval_batch_size = 10
# train_data = batchify(corpus.train, args.batch_size)
# val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    """
    if i == 0:
        data_words = [[""] * data.size(1)] * data.size(0)
        targets_words = [[""] * data.size(1)] * data.size(0)
        target_unflat = source[i+1:i+1+seq_len]
        for j in range(0, data.size(0)):
            for k in range(0, data.size(1)):
                data_words[j][k] = corpus.dictionary.idx2word[data[j][k]]
        for j in range(0, data.size(0)):
            for k in range(0, data.size(1)):
                targets_words[j][k] = corpus.dictionary.idx2word[target_unflat[j][k]]
        print("data words",data_words)
        print("targets words",targets_words)
        """
    return data, target

# call with bptt=1
# python sample.py --cuda --data ../../../visdial_text --model GRU
# --load model_gru_6.pt --bptt 1
def sample(data_source):
    print("in sample!!!!")
    # Turn on evaluation mode which disables dropout.
    model.eval()
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
      for data_idx in range(0, 100, 10):
        data, targets = get_batch(data_source, data_idx)
        samples = torch.zeros(20,10)
        samples[0,:] = data
        for i in range(1,20):
            output, hidden = model(data, hidden)
            topv, topi = output.topk(1,dim=2)
            # topi contains index of top predicted next word
            topi = topi[-1,:,:].view(1,10)
            # append to sampled sentences
            samples[i,:] = topi
            # set data to predicted next words for next iteration
            data = torch.cat((data,topi))
        for i in range(10):
            sample = ""
            for j in range(20):
                sample += (corpus.dictionary.idx2word[int(samples[j][i])] + " ")
            print(sample)
    return samples

# Load the model.
with open(args.load, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

sample(test_data)
