import argparse
import datetime
import gc
import json
import math
import os
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import VisDialDataset
from encoders import Encoder, EncoderParams
from decoders import Decoder
from utils import process_ranks, scores_to_ranks, get_gt_ranks, convert_to_string


parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)

parser.add_argument_group('Evaluation related arguments')
parser.add_argument('-load_path', default='checkpoints/model.pth',
                        help='Checkpoint to load path from')
parser.add_argument('-split', default='val', choices=['val', 'test'],
                        help='Split to evaluate on')
parser.add_argument('-use_gt', action='store_true',
                        help='Whether to use ground truth for retrieving ranks')
parser.add_argument('-batch_size', default=12, type=int, help='Batch size')
parser.add_argument('-gpuid', default=0, type=int, help='GPU id to use')
parser.add_argument('-overfit', action='store_true',
                        help='Use a batch of only 5 examples, useful for debugging')

parser.add_argument_group('Submission related arguments')
parser.add_argument('-save_ranks', action='store_true',
                        help='Whether to save retrieved ranks')
parser.add_argument('-save_path', default='logs/ranks.json',
                        help='Path of json file to save ranks')

# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()
if args.use_gt:
    if args.split == 'test':
        print("Warning: No ground truth for test split, changing use_gt to False.")
        args.use_gt = False
    elif args.split == 'val' and args.save_ranks:
        print("Warning: Cannot generate submission json if use_gt is True.")
        args.save_ranks = False

# seed for reproducibility
torch.manual_seed(1234)

# set device and default tensor type
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    torch.cuda.set_device(args.gpuid)

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------

components = torch.load(args.load_path)
model_args = components['model_args']
model_args.gpuid = args.gpuid
model_args.batch_size = args.batch_size
encoder_params = EncoderParams(model_args)

# this is required by dataloader
args.img_norm = model_args.img_norm

# set this because only late fusion encoder is supported yet
args.concat_history = encoder_params['concat_history']
args.partial_concat_history = encoder_params['partial_concat_history']

for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = VisDialDataset(args, [args.split])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=dataset.collate_fn)
ind2word = dataset.ind2word

# iterations per epoch
setattr(args, 'iter_per_epoch',
    math.ceil(dataset.num_data_points[args.split] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

encoder = Encoder(model_args)
encoder.load_state_dict(components['encoder'])

decoder = Decoder(model_args, encoder)
decoder.load_state_dict(components['decoder'])
print("Loaded model from {}".format(args.load_path))

if args.gpuid >= 0:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

print("Evaluation start time: {}".format(
    datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')))
encoder.eval()
decoder.eval()
if args.use_gt:
    # ------------------------------------------------------------------------
    # calculate automatic metrics and finish
    # ------------------------------------------------------------------------
    all_ranks = []
    all_labels = []
    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if not isinstance(batch[key], list):
                with torch.no_grad():
                    batch[key] = Variable(batch[key])
                    if args.gpuid >= 0:
                        batch[key] = batch[key].cuda()
        #print(batch['type'])
        enc_out = encoder(batch)
        dec_out = decoder(enc_out, batch)
        #print(dec_out[0])
        ranks = scores_to_ranks(dec_out.data)
        #print(ranks[0])
        gt_ranks = get_gt_ranks(ranks, batch['ans_ind'].data)
        #print(gt_ranks)
        
        batch_size = batch['ques'].size(0)
        round_length = batch['ques'].size(1)
       
        count = 0
        for b in range(batch_size):
            for r in range(round_length):
                ques_string = convert_to_string(batch['ques'][b][r], ind2word)
                ans_ind = batch['ans_ind'][b][r]
                gt_ans = convert_to_string(batch['opt'][b][r][ans_ind], ind2word)
                gt_rank = ranks[count][ans_ind]
                top_ranked_ind = None
                for ind in range(ranks.size(1)):
                    if ranks[count][ind] == 1:
                        top_ranked_ind = ind
                top_rank_ans = convert_to_string(batch['opt'][b][r][top_ranked_ind], ind2word)
                image_fname = batch['img_fnames'][b]
                if gt_rank > 1:
                    print("=====================\n%s\n%d %s\n%s\n%s" % (ques_string, gt_rank, gt_ans, top_rank_ans, image_fname))
                count += 1

        all_ranks.append(gt_ranks)
        for j in range(len(batch['type'])):
            for k in range(len(batch['type'][j])):
                all_labels.append(batch['type'][j][k])
             
    all_ranks = torch.cat(all_ranks, 0)
    #print (all_labels)
    yes_no_ranks = []
    color_ranks = []
    other_ranks = []
    count_ranks = []
    for j in range(len(all_ranks)):
        if (all_labels[j] == "yn"): 
            yes_no_ranks.append(all_ranks[j])
        if (all_labels[j] == "color"):
            color_ranks.append(all_ranks[j])
        if (all_labels[j] == "other"):
            other_ranks.append(all_ranks[j])
        if (all_labels[j] == "count"):
            count_ranks.append(all_ranks[j])

    yes_no_ranks = torch.tensor(yes_no_ranks)
    color_ranks = torch.tensor(color_ranks)
    count_ranks = torch.tensor(count_ranks)
    other_ranks = torch.tensor(other_ranks)
    process_ranks(all_ranks)
    print("Yes No stats")
    process_ranks(yes_no_ranks)
    print("Color stats")
    process_ranks(color_ranks)
    print("Count stats")
    process_ranks(count_ranks)
    print("Other stats")
    process_ranks(other_ranks)

    #for rank in all_ranks:
    #     print(rank)
    gc.collect()
else:
    # ------------------------------------------------------------------------
    # prepare json for submission
    # ------------------------------------------------------------------------
    ranks_json = []
    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if not isinstance(batch[key], list):
                with torch.no_grad():
                    batch[key] = Variable(batch[key])
                    if args.gpuid >= 0:
                        batch[key] = batch[key].cuda()

        enc_out = encoder(batch)
        dec_out = decoder(enc_out, batch)
        ranks = scores_to_ranks(dec_out.data)
        ranks = ranks.view(-1, 10, 100)
        ranks = ranks.to(torch.device("cpu"))
        ranks = ranks.numpy().tolist()
 
        for i in range(len(batch['img_fnames'])):
            # cast into types explicitly to ensure no errors in schema
            if args.split == 'test':
               ranks_json.append({
                    'image_id': int(batch['img_fnames'][i][-16:-4]),
                    'round_id': int(batch['num_rounds'][i]),
                    'ranks': list(ranks[i][batch['num_rounds'][i] - 1])
                })
            else:
                for j in range(batch['num_rounds'][i]):
                    ranks_json.append({
                        'image_id': int(batch['img_fnames'][i][-16:-4]),
                        'round_id': int(j + 1),
                        'ranks': list(ranks[i][j])
                    })
        gc.collect()

if args.save_ranks:
    print("Writing ranks to {}".format(args.save_path))
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    json.dump(ranks_json, open(args.save_path, 'w'))
