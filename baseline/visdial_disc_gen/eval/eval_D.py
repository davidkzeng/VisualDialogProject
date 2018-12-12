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

from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt
import misc.dataLoader as dl
import misc.model as model
from misc.encoder_QIH import _netE
from misc.encoder_MCB_QIH import _netE as _netE_MCB
import datetime
import h5py
import wmd
import spacy
from num2words import num2words
from metrics.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from metrics.pycocoevalcap.bleu.bleu import Bleu
from metrics.pycocoevalcap.rouge.rouge import Rouge

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='', help='folder to output images and model checkpoints')
parser.add_argument('--input_img_h5', default='vdl_img_vgg.h5', help='')
parser.add_argument('--input_ques_h5', default='visdial_data.h5', help='visdial_data.h5')
parser.add_argument('--input_json', default='visdial_params.json', help='visdial_params.json')

parser.add_argument('--model_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--model_mcb_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--mcb'  , action='store_true', help='uses mcb instead of concatenation')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####################################################################################
# Data Loader
####################################################################################

if opt.model_path != '':
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path)
    model_path = opt.model_path
    model_mcb_path = opt.model_mcb_path
    data_dir = opt.data_dir
    input_img_h5 = opt.input_img_h5
    input_ques_h5 = opt.input_ques_h5
    input_json = opt.input_json
    print("opt.input_json", opt.input_json)
    old_mcb = opt.mcb
    opt = checkpoint['opt']
    opt.mcb = old_mcb
    opt.start_epoch = checkpoint['epoch']
    opt.batchSize = 5
    opt.data_dir = data_dir
    opt.model_path = model_path
    opt.model_mcb_path = model_mcb_path

if opt.model_mcb_path != '':
    print("=> loading mcb checkpoint '{}'".format(opt.model_mcb_path))
    mcb_checkpoint = torch.load(opt.model_mcb_path) # ???

input_img_h5 = os.path.join(opt.data_dir, opt.input_img_h5)
input_ques_h5 = os.path.join(opt.data_dir, opt.input_ques_h5)
input_json = os.path.join(opt.data_dir, opt.input_json)
print("opt.input_json", opt.input_json)
print("input_json file", input_json)

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
netE_MCB = _netE_MCB(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout,
    img_feat_size)

netW = model._netW(n_words, opt.ninp, opt.dropout)
netW_MCB = model._netW(n_words, opt.ninp, opt.dropout)
netD = model._netD(opt.model, opt.ninp, opt.nhid, opt.nlayers, n_words, opt.dropout)
netD_MCB = model._netD(opt.model, opt.ninp, opt.nhid, opt.nlayers, n_words, opt.dropout)

if opt.model_path != '': # load the pre-trained model.
    netW.load_state_dict(checkpoint['netW'])
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])
    print('Loading model Success!')

if opt.model_mcb_path != '': # load the pre-trained MCB model
    netW_MCB.load_state_dict(mcb_checkpoint['netW'])
    netE_MCB.load_state_dict(mcb_checkpoint['netE_MCB'])
    netD_MCB.load_state_dict(mcb_checkpoint['netD'])
    print('Loading MCB model Success!')

if opt.cuda: # ship to cuda, if has GPU
    netW.cuda(), netE.cuda(), netD.cuda()
    netW_MCB.cuda(), netE_MCB.cuda(), netD_MCB.cuda()

# Word Mover's Distance model
wmd_nlp = spacy.load('en_core_web_lg')
wmd_nlp.add_pipe(wmd.WMD.SpacySimilarityHook(wmd_nlp), last=True)
# spaCy English model
nlp = spacy.load('en_core_web_lg')

####################################################################################
# Some Functions
####################################################################################

def eval():
    netW.eval()
    netE.eval()
    netD.eval()

    netW_MCB.eval()
    netE_MCB.eval()
    netD_MCB.eval()

    data_iter_val = iter(dataloader_val)
    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)

    ques_hidden_mcb = netE_MCB.init_hidden(opt.batchSize)
    hist_hidden_mcb = netE_MCB.init_hidden(opt.batchSize)

    opt_hidden = netD.init_hidden(opt.batchSize)
    opt_hidden_mcb = netD_MCB.init_hidden(opt.batchSize)
    i = 0
    total_count = 0
    total_sim = 0
    total_sim_mcb = 0
    total_wmd = 0
    total_wmd_mcb = 0
    wmd_count = 0
    total_metrics = {}
    total_metrics_mcb = {}
    total_metrics_count = {}
    total_metrics_count_mcb = {}
    tokenizer = PTBTokenizer()
    scorers = [ (Bleu(4), "Bleu4"),
                (Rouge(), "ROUGE_L")]
    for scorer in scorers:
        total_metrics[scorer[1]] = 0
        total_metrics_mcb[scorer[1]] = 0
        total_metrics_count[scorer[1]] = 0
        total_metrics_count_mcb[scorer[1]] = 0
    rank_all_tmp = []
    rank_all_tmp_mcb = []
    img_atten = torch.FloatTensor(100 * 30, 10, 7, 7)
    rank_color = []
    rank_count = []
    rank_yn = []
    rank_round = [[], [], [], [], [], [], [], [], [], []]
    rank_color_mcb = []
    rank_count_mcb  = []
    rank_yn_mcb  = []
    rank_round_mcb  = [[], [], [], [], [], [], [], [], [], []]
    while i < 3000:#len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                opt_answerT, answer_ids, answerLen, opt_answerLen, img_id  = data
        img_id_data = img_id.data.cpu().numpy()

        batch_size = question.size(0)
        image = image.view(-1, 512)
        img_input.data.resize_(image.size()).copy_(image)

        gts = {}
        trs = {}
        trs_mcb = {}
        for rnd in range(10):
            # get the corresponding round QA and history.
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

            opt_ans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            gt_id = answer_ids[:,rnd]

            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)

            gt_index.data.resize_(gt_id.size()).copy_(gt_id)
            opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)

            opt_len = opt_answerLen[:,rnd,:].clone().view(-1)

            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_emb_mcb = netW_MCB(ques_input, format = 'index')
            his_emb_mcb = netW_MCB(his_input, format = 'index')

            ques_hidden = repackage_hidden(ques_hidden, batch_size)
            hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))

            ques_hidden_mcb = repackage_hidden(ques_hidden_mcb, batch_size)
            hist_hidden_mcb = repackage_hidden(hist_hidden_mcb, his_input.size(1))

            featD, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                                ques_hidden, hist_hidden, rnd+1)
            featD_mcb, ques_hidden_mcb = netE_MCB(ques_emb_mcb, his_emb_mcb, img_input, \
                                                ques_hidden_mcb, hist_hidden_mcb, rnd+1)

            opt_ans_emb = netW(opt_ans_input, format = 'index')
            opt_hidden = repackage_hidden(opt_hidden, opt_ans_input.size(1))
            opt_feat = netD(opt_ans_emb, opt_ans_input, opt_hidden, n_words)
            opt_feat = opt_feat.view(batch_size, -1, opt.ninp)

            opt_ans_emb_mcb = netW_MCB(opt_ans_input, format = 'index')
            opt_hidden_mcb = repackage_hidden(opt_hidden_mcb, opt_ans_input.size(1))
            opt_feat_mcb = netD_MCB(opt_ans_emb_mcb, opt_ans_input, opt_hidden_mcb, n_words)
            opt_feat_mcb = opt_feat_mcb.view(batch_size, -1, opt.ninp)

            featD = featD.view(-1, opt.ninp, 1)
            score = torch.bmm(opt_feat, featD)
            score = score.view(-1, 100)

            featD_mcb = featD_mcb.view(-1, opt.ninp, 1)
            score_mcb = torch.bmm(opt_feat_mcb, featD_mcb)
            score_mcb = score_mcb.view(-1, 100)

            for b in range(batch_size):
                gt_index.data[b] = gt_index.data[b] + b*100

            gt_score = score.view(-1).index_select(0, gt_index)
            sort_score, sort_idx = torch.sort(score, 1, descending=True)

            gt_score_mcb = score_mcb.view(-1).index_select(0, gt_index)
            sort_score_mcb, sort_idx_mcb = torch.sort(score_mcb, 1, descending=True)

            count = sort_score.gt(gt_score.view(-1,1).expand_as(sort_score))
            rank = count.sum(1) + 1
            rank_all_tmp += list(rank.view(-1).data.cpu().numpy())

            count_mcb = sort_score_mcb.gt(gt_score_mcb.view(-1,1).expand_as(sort_score_mcb))
            rank_mcb = count_mcb.sum(1) + 1
            rank_all_tmp_mcb += list(rank_mcb.view(-1).data.cpu().numpy())

            """
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
                print("sort_idx size",sort_idx.size())
            """

            ques_txt = decode_txt(itow, questionL[:,rnd,:].t())
            rank_data = rank.data.cpu().numpy()
            rank_round[rnd] += list(rank_data)
            rank_data_mcb = rank_mcb.data.cpu().numpy()
            rank_round_mcb[rnd] += list(rank_data_mcb)
            for j in range(batch_size):
                tmp_tans = opt_answerT[j,rnd,int(gt_id[j]),:]
                top_ans = opt_answerT[j,rnd,int(sort_idx[j][0]),:]
                tans_txt = decode_txt(itow, tmp_tans.view(tmp_tans.size()[0],1))
                top_ans_txt = decode_txt(itow, top_ans.view(top_ans.size()[0],1))

                top_ans_mcb = opt_answerT[j,rnd,int(sort_idx_mcb[j][0]),:]
                top_ans_txt_mcb = decode_txt(itow, top_ans_mcb.view(top_ans_mcb.size()[0],1))

                if "yes" in tans_txt[0] or "no" in tans_txt[0]:
                    rank_yn += [rank_data[j]]
                    rank_yn_mcb += [rank_data_mcb[j]]
                elif "color" in ques_txt[j]:
                    rank_color += [rank_data[j]]
                    rank_color_mcb += [rank_data_mcb[j]]
                elif "how many" in ques_txt[j] or any(char.isdigit() for char in tans_txt[0]):
                    rank_count += [rank_data[j]]
                    rank_count_mcb += [rank_data_mcb[j]]

                if top_ans_txt[0] != top_ans_txt_mcb[0]:
                    print('rnd: %d img_id: %d Q: %s --A: %s'
                        %(rnd, img_id_data[j], ques_txt[j], tans_txt[0]))
                    print('Sampled from candidate answers: %s rank of A: %d'
                        %(top_ans_txt[0], rank_data[j]))
                    print('MCB Sampled from candidate answers: %s rank of A: %d'
                        %(top_ans_txt_mcb[0], rank_data_mcb[j]))

                gt_ans = tans_txt[0]
                top_rank_ans = top_ans_txt[0]
                gt_doc = nlp(gt_ans)
                top_rank_doc = nlp(top_rank_ans)
                sim = gt_doc.similarity(top_rank_doc)
                gts[b * rnd + rnd] = [{ u'caption' : gt_ans }]
                trs[b * rnd + rnd] = [{ u'caption' : top_rank_ans }]
                total_sim += sim
                total_count += 1

                top_rank_ans_mcb = top_ans_txt_mcb[0]
                top_rank_doc_mcb = nlp(top_rank_ans_mcb)
                sim_mcb = gt_doc.similarity(top_rank_doc_mcb)
                trs_mcb[b * rnd + rnd] = [{ u'caption' : top_rank_ans_mcb }]
                total_sim_mcb += sim_mcb
                """
                if i % 10 == 0 and j == 0:
                  print("question, gt_ans, top_rank_ans, cosine sim",
                      ques_txt[j], gt_ans,
                      top_rank_ans, sim)
                """

                # Converting numbers to their word representations
                gt_ans_temp = gt_ans
                gt_ans = ""
                for word in gt_ans_temp.split():
                    if (word.isdigit()):
                        gt_ans = gt_ans + num2words(int(word)) + " "
                    else:
                        gt_ans = gt_ans + word + " "

                top_rank_ans_temp = top_rank_ans
                top_rank_ans = ""
                for word in top_rank_ans_temp.split():
                    if (word.isdigit()):
                        top_rank_ans = top_rank_ans + num2words(int(word)) + " "
                    else:
                        top_rank_ans = top_rank_ans + word + " "

                top_rank_ans_temp_mcb = top_rank_ans_mcb
                top_rank_ans_mcb = ""
                for word in top_rank_ans_temp_mcb.split():
                    if (word.isdigit()):
                        top_rank_ans_mcb = top_rank_ans_mcb + num2words(int(word)) + " "
                    else:
                        top_rank_ans_mcb = top_rank_ans_mcb + word + " "

                if (gt_ans == "no- "):
                    gt_ans = "no"
                if (top_rank_ans == "no- "):
                    top_rank_ans = "no"
                if (top_rank_ans_mcb == "no- "):
                    top_rank_ans_mcb = "no"

                # wmd implementation does not allow for digits within answers so they are excluded from calculations
                if (not(any(char.isdigit() for char in gt_ans) or any(char.isdigit() for char in top_rank_ans))):
                    wmd_gt_doc = wmd_nlp(gt_ans)
                    wmd_top_rank_doc = wmd_nlp(top_rank_ans)
                    try:
                        wmd_sim = wmd_gt_doc.similarity(wmd_top_rank_doc)
                    except RuntimeError as err:
                        print("gt_ans, top_rank_ans", gt_ans,
                          top_rank_ans)
                        print("WMD similarity runtime error",err)
                    """
                    if i % 10 == 0 and j == 0:
                      print("gt_ans, top_rank_ans, wmd sim", gt_ans,
                          top_rank_ans, wmd_sim)
                    """
                    total_wmd += wmd_sim
                    wmd_count += 1

                if (not(any(char.isdigit() for char in gt_ans) or
                  any(char.isdigit() for char in top_rank_ans_mcb))):
                    wmd_gt_doc = wmd_nlp(gt_ans)
                    wmd_top_rank_doc_mcb = wmd_nlp(top_rank_ans_mcb)
                    try:
                        wmd_sim_mcb = wmd_gt_doc.similarity(wmd_top_rank_doc_mcb)
                    except RuntimeError as err:
                        print("gt_ans, top_rank_ans_mcb", gt_ans,
                          top_rank_ans_mcb)
                        print("WMD similarity runtime error",err)
                    """
                    if i % 10 == 0 and j == 0:
                      print("gt_ans, top_rank_ans, wmd sim", gt_ans,
                          top_rank_ans, wmd_sim)
                    """
                    total_wmd_mcb += wmd_sim_mcb

                # print('Q: %s --A: %s' % (ques_txt[j], tans_txt[0]))

        gt_tokens = tokenizer.tokenize(gts)
        tr_tokens = tokenizer.tokenize(trs)
        tr_tokens_mcb = tokenizer.tokenize(trs_mcb)
        for scorer_name in scorers:
            scorer, name = scorer_name
            _, scores = scorer.compute_score(gt_tokens, tr_tokens)
            _, scores_mcb = scorer.compute_score(gt_tokens, tr_tokens_mcb)
            if name == "Bleu4":
                scores = scores[0]
                scores_mcb = scores_mcb[0]
            for score in scores:
                total_metrics[name] += score
                total_metrics_count[name] += 1
            for score in scores_mcb:
                total_metrics_mcb[name] += score
                total_metrics_count_mcb[name] += 1
        i += 1

        if i % 20 == 0 or i == len(dataloader_val):
            R1 = np.sum(np.array(rank_all_tmp)==1) / float(len(rank_all_tmp))
            R5 =  np.sum(np.array(rank_all_tmp)<=5) / float(len(rank_all_tmp))
            R10 = np.sum(np.array(rank_all_tmp)<=10) / float(len(rank_all_tmp))
            ave = np.sum(np.array(rank_all_tmp)) / float(len(rank_all_tmp))
            mrr = np.sum(1/(np.array(rank_all_tmp, dtype='float'))) / float(len(rank_all_tmp))
            print ('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(i, len(dataloader_val), mrr, R1, R5, R10, ave))

            R1 = np.sum(np.array(rank_all_tmp_mcb)==1) / float(len(rank_all_tmp_mcb))
            R5 =  np.sum(np.array(rank_all_tmp_mcb)<=5) / float(len(rank_all_tmp_mcb))
            R10 = np.sum(np.array(rank_all_tmp_mcb)<=10) / float(len(rank_all_tmp_mcb))
            ave = np.sum(np.array(rank_all_tmp_mcb)) / float(len(rank_all_tmp_mcb))
            mrr = np.sum(1/(np.array(rank_all_tmp_mcb, dtype='float'))) / float(len(rank_all_tmp_mcb))
            print ('%d/%d MCB: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(i, len(dataloader_val), mrr, R1, R5, R10, ave))

            avg_sim = total_sim / total_count
            avg_wmd = total_wmd / wmd_count
            print ('%d/%d: Average similarity: %f Average Word Movers Distance: %f' %(i, len(dataloader_val), avg_sim, avg_wmd))
            for k, v in total_metrics.items():
                v_avg = v / total_metrics_count[k]
                print("Average %s %.5f" % (k, v_avg))

            avg_sim = total_sim_mcb / total_count
            avg_wmd = total_wmd_mcb / wmd_count
            print ('%d/%d MCB: Average similarity: %f Average Word Movers Distance: %f' %(i, len(dataloader_val), avg_sim, avg_wmd))
            for k, v in total_metrics_mcb.items():
                v_avg = v / total_metrics_count_mcb[k]
                print("MCB Average %s %.5f" % (k, v_avg))

            for (name,rank_tmp) in [("color",rank_color),("count",rank_count),("yes/no",rank_yn)]:
                R1 = np.sum(np.array(rank_tmp)==1) / float(len(rank_tmp))
                R5 =  np.sum(np.array(rank_tmp)<=5) / float(len(rank_tmp))
                R10 = np.sum(np.array(rank_tmp)<=10) / float(len(rank_tmp))
                ave = np.sum(np.array(rank_tmp)) / float(len(rank_tmp))
                mrr = np.sum(1/(np.array(rank_tmp, dtype='float'))) / float(len(rank_tmp))
                print ('%d/%d %s (%d out of %d): mrr: %f R1: %f R5 %f R10 %f Mean %f' %(i,
                    len(dataloader_val), name, len(rank_tmp), i*50, mrr, R1, R5, R10, ave))

            for (name,rank_tmp) in [("color",rank_color_mcb),("count",rank_count_mcb),("yes/no",rank_yn_mcb)]:
                R1 = np.sum(np.array(rank_tmp)==1) / float(len(rank_tmp))
                R5 =  np.sum(np.array(rank_tmp)<=5) / float(len(rank_tmp))
                R10 = np.sum(np.array(rank_tmp)<=10) / float(len(rank_tmp))
                ave = np.sum(np.array(rank_tmp)) / float(len(rank_tmp))
                mrr = np.sum(1/(np.array(rank_tmp, dtype='float'))) / float(len(rank_tmp))
                print ('%d/%d %s (%d out of %d) MCB: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(i,
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

            for (r,rank_tmp) in enumerate(rank_round_mcb):
                if len(rank_tmp) > 0:
                    R1 = np.sum(np.array(rank_tmp)==1) / float(len(rank_tmp))
                    R5 =  np.sum(np.array(rank_tmp)<=5) / float(len(rank_tmp))
                    R10 = np.sum(np.array(rank_tmp)<=10) / float(len(rank_tmp))
                    ave = np.sum(np.array(rank_tmp)) / float(len(rank_tmp))
                    mrr = np.sum(1/(np.array(rank_tmp, dtype='float'))) / float(len(rank_tmp))
                    print ('%d/%d, round %d MCB: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(i,
                        len(dataloader_val), r, mrr, R1, R5, R10, ave))

    return img_atten

####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize)
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)

# answer input
opt_ans_input = torch.LongTensor(ans_length, opt.batchSize)
fake_ans_input = torch.FloatTensor(ques_length, opt.batchSize, n_words)
sample_ans_input = torch.LongTensor(1, opt.batchSize)

# answer index location.
opt_index = torch.LongTensor( opt.batchSize)
fake_index = torch.LongTensor(opt.batchSize)

batch_sample_idx = torch.LongTensor(opt.batchSize)
# answer len
fake_len = torch.LongTensor(opt.batchSize)

# noise
noise_input = torch.FloatTensor(opt.batchSize)
gt_index = torch.LongTensor(opt.batchSize)


if opt.cuda:
    ques_input, his_input, img_input = ques_input.cuda(), his_input.cuda(), img_input.cuda()
    opt_ans_input = opt_ans_input.cuda()
    fake_ans_input, sample_ans_input = fake_ans_input.cuda(), sample_ans_input.cuda()
    opt_index, fake_index =  opt_index.cuda(), fake_index.cuda()

    fake_len = fake_len.cuda()
    noise_input = noise_input.cuda()
    batch_sample_idx = batch_sample_idx.cuda()
    gt_index = gt_index.cuda()


ques_input = Variable(ques_input)
img_input = Variable(img_input)
his_input = Variable(his_input)

opt_ans_input = Variable(opt_ans_input)
fake_ans_input = Variable(fake_ans_input)
sample_ans_input = Variable(sample_ans_input)

opt_index = Variable(opt_index)
fake_index = Variable(fake_index)

fake_len = Variable(fake_len)
noise_input = Variable(noise_input)
batch_sample_idx = Variable(batch_sample_idx)
gt_index = Variable(gt_index)

atten = eval()
