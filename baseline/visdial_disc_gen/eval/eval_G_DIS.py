from __future__ import print_function
import argparse
import os
import random
import sys
sys.path.append(os.getcwd())

import pdb
import time
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, \
                    decode_txt, sample_batch_neg, l2_norm
import misc.dataLoader as dl
import misc.model as model
from misc.encoder_QIH import _netE
import datetime
from misc.netG import _netG

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='', help='folder to output images and model checkpoints')
parser.add_argument('--input_img_h5', default='vdl_img_vgg.h5', help='')
parser.add_argument('--input_ques_h5', default='visdial_data.h5', help='visdial_data.h5')
parser.add_argument('--input_json', default='visdial_params.json', help='visdial_params.json')

parser.add_argument('--model_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')

opt = parser.parse_args()

opt.manualSeed = random.randint(1, 10000) # fix seed
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.model_path != '':
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path)
    model_path = opt.model_path
    data_dir = opt.data_dir
    input_img_h5 = opt.input_img_h5
    input_ques_h5 = opt.input_ques_h5
    input_json = opt.input_json
    opt = checkpoint['opt']
    opt.start_epoch = checkpoint['epoch']
    opt.batchSize = 5
    opt.data_dir = data_dir
    opt.model_path = model_path

####################################################################################
# Data Loader
####################################################################################

input_img_h5 = os.path.join(opt.data_dir, opt.input_img_h5)
input_ques_h5 = os.path.join(opt.data_dir, opt.input_ques_h5)
input_json = os.path.join(opt.data_dir, opt.input_json)

dataset_val = dl.validate(input_img_h5=input_img_h5, input_ques_h5=input_ques_h5,
                input_json=input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'test')


dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
####################################################################################
# Build the Model
####################################################################################

n_words = dataset_val.vocab_size
ques_length = dataset_val.ques_length
ans_length = dataset_val.ans_length + 1
his_length = ques_length+dataset_val.ans_length
itow = dataset_val.itow
img_feat_size = 512

netE = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)
netW = model._netW(n_words, opt.ninp, opt.dropout)
netG = _netG(opt.model, n_words, opt.ninp, opt.nhid, opt.nlayers, opt.dropout)
critG = model.LMCriterion()
sampler = model.gumbel_sampler()


if opt.cuda:
    netW.cuda()
    netE.cuda()
    netG.cuda()
    critG.cuda()
    sampler.cuda()

if opt.model_path != '':
    netW.load_state_dict(checkpoint['netW_g'])
    netE.load_state_dict(checkpoint['netE_g'])
    netG.load_state_dict(checkpoint['netG'])
    print('Loading model Success!')

def eval():

    netE.eval()
    netW.eval()
    netG.eval()

    data_iter_val = iter(dataloader_val)
    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)

    i = 0
    display_count = 0
    average_loss = 0
    rank_all_tmp = []
    while i < len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                opt_answerT, answer_ids, answerLen, opt_answerLen, img_id  = data

        batch_size = question.size(0)
        image = image.view(-1, 512)
        img_input.data.resize_(image.size()).copy_(image)

        for rnd in range(10):
            # get the corresponding round QA and history.
            ques, tans = question[:,rnd,:].t(), opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()
            ans = opt_answer[:,rnd,:,:].clone().view(-1, ans_length).t()
            gt_id = answer_ids[:,rnd]

            his_input.data.resize_(his.size()).copy_(his)
            ques_input.data.resize_(ques.size()).copy_(ques)
            ans_input.data.resize_(ans.size()).copy_(ans)
            ans_target.data.resize_(tans.size()).copy_(tans)

            gt_index.data.resize_(gt_id.size()).copy_(gt_id)

            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')


            ques_hidden = repackage_hidden(ques_hidden, batch_size)
            hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))

            encoder_feat, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                                ques_hidden, hist_hidden, rnd+1)

            _, ques_hidden = netG(encoder_feat.view(1,-1,opt.ninp), ques_hidden)

            #ans_emb = ans_emb.view(ans_length, -1, 100, opt.nhid)
            ans_score = torch.FloatTensor(batch_size, 100).zero_()
            # extend the hidden
            hidden_replicated = []
            for hid in ques_hidden:
                hidden_replicated.append(hid.view(opt.nlayers, batch_size, 1, \
                    opt.nhid).expand(opt.nlayers, batch_size, 100, opt.nhid).clone().view(opt.nlayers, -1, opt.nhid))
            hidden_replicated = tuple(hidden_replicated)

            ans_emb = netW(ans_input, format = 'index')

            output, _ = netG(ans_emb, hidden_replicated)
            logprob = - output
            logprob_select = torch.gather(logprob, 1, ans_target.view(-1,1))

            mask = ans_target.data.eq(0)  # generate the mask
            if isinstance(logprob, Variable):
                mask = Variable(mask, volatile=logprob.volatile)
            logprob_select.masked_fill_(mask.view_as(logprob_select), 0)

            prob = logprob_select.view(ans_length, -1, 100).sum(0).view(-1,100)

            for b in range(batch_size):
                gt_index.data[b] = gt_index.data[b] + b*100

            gt_score = prob.view(-1).index_select(0, gt_index)
            sort_score, sort_idx = torch.sort(prob, 1)

            count = sort_score.lt(gt_score.view(-1,1).expand_as(sort_score))
            rank = count.sum(1) + 1
            rank_all_tmp += list(rank.view(-1).data.cpu().numpy())

            if i == 0 and rnd == 0:
                print("gt_score size",gt_score.size())
                print("gt_score",gt_score)
                print("sort_score size",sort_score.size())
                print("sort_score",sort_score)
                print("count size",count.size())
                print("count",count)
                print("rank size",rank.view(-1).size())
                print("rank",rank.view(-1))
                print("tans size", opt_answerT[:,rnd,:,:].clone().view(-1,
                    ans_length).t().size())
                print("opt_answerT size", opt_answerT.size())

            """
            ques_txt = decode_txt(itow, questionL[:,rnd,:].t())
            for j in range(batch_size):
                tmp_tans = opt_answerT[j,rnd,int(gt_id[j]),:]
                tans_txt = decode_txt(itow, tmp_tans.view(tmp_tans.size()[0],1))
                print('Q: %s --A: %s' % (ques_txt[j], tans_txt[0]))
            """

        i += 1
        sys.stdout.write('Evaluating: {:d}/{:d}  \r' \
          .format(i, len(dataloader_val)))

        if i % 50 == 0:
            R1 = np.sum(np.array(rank_all_tmp)==1) / float(len(rank_all_tmp))
            R5 =  np.sum(np.array(rank_all_tmp)<=5) / float(len(rank_all_tmp))
            R10 = np.sum(np.array(rank_all_tmp)<=10) / float(len(rank_all_tmp))
            ave = np.sum(np.array(rank_all_tmp)) / float(len(rank_all_tmp))
            mrr = np.sum(1/(np.array(rank_all_tmp, dtype='float'))) / float(len(rank_all_tmp))
            print ('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(i, len(dataloader_val), mrr, R1, R5, R10, ave))

    return rank_all_tmp


def sample():
    netE.eval()
    netW.eval()
    netG.eval()

    data_iter_val = iter(dataloader_val)
    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)

    i = 0
    rank_color = []
    rank_count = []
    rank_yn = []
    rank_round = [[], [], [], [], [], [], [], [], [], []]
    while i < len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                opt_answerT, answer_ids, answerLen, opt_answerLen, img_id  = data
        img_id_data = img_id.data.cpu().numpy()

        batch_size = question.size(0)
        image = image.view(-1, 512)
        img_input.data.resize_(image.size()).copy_(image)

        for rnd in range(10):
            # get the corresponding round QA and history.
            ques, tans = question[:,rnd,:].t(), opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()
            ans = opt_answer[:,rnd,:,:].clone().view(-1, ans_length).t()
            gt_id = answer_ids[:,rnd]

            his_input.data.resize_(his.size()).copy_(his)
            ques_input.data.resize_(ques.size()).copy_(ques)
            ans_input.data.resize_(ans.size()).copy_(ans)
            ans_target.data.resize_(tans.size()).copy_(tans)

            gt_index.data.resize_(gt_id.size()).copy_(gt_id)

            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_hidden = repackage_hidden(ques_hidden, batch_size)
            hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))

            encoder_feat, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                                ques_hidden, hist_hidden, rnd+1)
            _, ques_hidden = netG(encoder_feat.view(1,-1,opt.ninp), ques_hidden)

            hidden_replicated = []
            for hid in ques_hidden:
                hidden_replicated.append(hid.view(opt.nlayers, batch_size, 1, \
                    opt.nhid).expand(opt.nlayers, batch_size, 100, opt.nhid).clone().view(opt.nlayers, -1, opt.nhid))
            hidden_replicated = tuple(hidden_replicated)

            ans_emb = netW(ans_input, format = 'index')

            output, _ = netG(ans_emb, hidden_replicated)
            logprob = - output
            logprob_select = torch.gather(logprob, 1, ans_target.view(-1,1))

            mask = ans_target.data.eq(0)  # generate the mask
            if isinstance(logprob, Variable):
                mask = Variable(mask, volatile=logprob.volatile)
            logprob_select.masked_fill_(mask.view_as(logprob_select), 0)

            prob = logprob_select.view(ans_length, -1, 100).sum(0).view(-1,100)

            for b in range(batch_size):
                gt_index.data[b] = gt_index.data[b] + b*100

            gt_score = prob.view(-1).index_select(0, gt_index)
            sort_score, sort_idx = torch.sort(prob, 1)
            count = sort_score.lt(gt_score.view(-1,1).expand_as(sort_score))
            rank = count.sum(1) + 1

            # ques_emb = embeder(ques_input)
            # hidden = repackage_hidden(hidden, batch_size)
            # _, hidden = encoder(ques_emb, hidden)

            #output, ques_hidden = decoder(ans_emb, ques_hidden)
            #loss = crit(output, ans_target.view(-1,1))
            #average_loss += loss.data[0]
            #count += 1
            ans_sample_result = torch.Tensor(ans_length, batch_size)
            ans_sample.data.resize_((1, batch_size)).fill_(n_words)
            # sample the result.


            noise_input.data.resize_(ans_length, batch_size, n_words+1)
            noise_input.data.uniform_(0,1)
            for t in range(ans_length):
                ans_sample_embed = netW(ans_sample, format = 'index')
                output, ques_hidden = netG(ans_sample_embed, ques_hidden)

                prob = - output
                #_, idx = torch.max(prob, 1)
                one_hot, idx = sampler(prob, noise_input[t], 0.5)

                ans_sample.data.copy_(idx.data)
                ans_sample_result[t].copy_(ans_sample.view(batch_size).data)

            # print("ans_sample_result", ans_sample_result)
            # print("ans_sample_result size", ans_sample_result.size())
            # print("tans size", tans.size())
            # print("gt_id size", gt_id.size())
            # print("ques size", questionL[:,rnd,:].t().size())
            ans_sample_txt = decode_txt(itow, ans_sample_result)
            ques_txt = decode_txt(itow, questionL[:,rnd,:].t())
            rank_data = rank.data.cpu().numpy()
            rank_round[rnd] += list(rank_data)
            for j in range(batch_size):
                tmp_tans = opt_answerT[j,rnd,int(gt_id[j]),:]
                top_ans = opt_answerT[j,rnd,int(sort_idx[j][0]),:]
                tans_txt = decode_txt(itow, tmp_tans.view(tmp_tans.size()[0],1))
                top_ans_txt = decode_txt(itow, top_ans.view(top_ans.size()[0],1))

                if "yes" in tans_txt[0] or "no" in tans_txt[0]:
                    rank_yn += [rank_data[j]]
                elif "color" in ques_txt[j]:
                    rank_color += [rank_data[j]]
                elif "how many" in ques_txt[j] or any(char.isdigit() for char in tans_txt[0]):
                    rank_count += [rank_data[j]]

                # if i % 25 == 0:
                #     print('rnd: %d img_id: %d Q: %s --A: %s -- Sampled from candidate answers: %s rank of A: %d, Sampled: %s'
                #         %(rnd, img_id_data[j], ques_txt[j], tans_txt[0], top_ans_txt[0], rank_data[j], ans_sample_txt[j]))

            # pdb.set_trace()
        i += 1
        sys.stdout.write('Evaluating: {:d}/{:d}  \r' \
          .format(i, len(dataloader_val)))

        if i % 50 == 0:
            for (name,rank_tmp) in [("color",rank_color),("count",rank_count),("yes/no",rank_yn)]:
                R1 = np.sum(np.array(rank_tmp)==1) / float(len(rank_tmp))
                R5 =  np.sum(np.array(rank_tmp)<=5) / float(len(rank_tmp))
                R10 = np.sum(np.array(rank_tmp)<=10) / float(len(rank_tmp))
                ave = np.sum(np.array(rank_tmp)) / float(len(rank_tmp))
                mrr = np.sum(1/(np.array(rank_tmp, dtype='float'))) / float(len(rank_tmp))
                print ('%d/%d %s (%d out of %d): mrr: %f R1: %f R5 %f R10 %f Mean %f' %(i,
                    len(dataloader_val), name, len(rank_tmp), i*50, mrr, R1, R5, R10, ave))
            for (r,rank_tmp) in enumerate(rank_round):
                if len(rank_tmp) > 0:
                    R1 = np.sum(np.array(rank_tmp)==1) / float(len(rank_tmp))
                    R5 =  np.sum(np.array(rank_tmp)<=5) / float(len(rank_tmp))
                    R10 = np.sum(np.array(rank_tmp)<=10) / float(len(rank_tmp))
                    ave = np.sum(np.array(rank_tmp)) / float(len(rank_tmp))
                    mrr = np.sum(1/(np.array(rank_tmp, dtype='float'))) / float(len(rank_tmp))
                    print ('%d/%d, round %d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(i,
                        len(dataloader_val), r, mrr, R1, R5, R10, ave))

    return rank_color, rank_count, rank_yn, rank_round

####################################################################################
# Main
####################################################################################
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)
img_input = torch.FloatTensor(opt.batchSize)

ans_input = torch.LongTensor(ans_length, opt.batchSize)
ans_target = torch.LongTensor(ans_length, opt.batchSize)
ans_sample = torch.LongTensor(1, opt.batchSize)

noise_input = torch.FloatTensor(opt.batchSize)
gt_index = torch.LongTensor(opt.batchSize)

if opt.cuda:
    ques_input, ans_input = ques_input.cuda(), ans_input.cuda()
    ans_target, ans_sample = ans_target.cuda(), ans_sample.cuda()
    gt_index = gt_index.cuda()
    noise_input = noise_input.cuda()
    his_input,img_input = his_input.cuda(),img_input.cuda()

ques_input = Variable(ques_input, volatile=True)
ans_input = Variable(ans_input, volatile=True)
ans_target = Variable(ans_target, volatile=True)
ans_sample = Variable(ans_sample, volatile=True)
noise_input = Variable(noise_input, volatile=True)
gt_index = Variable(gt_index, volatile = True)
his_input = Variable(his_input)
img_input = Variable(img_input)

# sample()

rank_all = eval()
R1 = np.sum(np.array(rank_all)==1) / float(len(rank_all))
R5 =  np.sum(np.array(rank_all)<=5) / float(len(rank_all))
R10 = np.sum(np.array(rank_all)<=10) / float(len(rank_all))
ave = np.sum(np.array(rank_all)) / float(len(rank_all))
mrr = np.sum(1/(np.array(rank_all, dtype='float'))) / float(len(rank_all))
print ('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(1, len(dataloader_val), mrr, R1, R5, R10, ave))
