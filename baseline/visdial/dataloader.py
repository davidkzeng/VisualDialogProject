import os
import json
import spacy
from six import iteritems
import collections

import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import convert_to_string


class VisDialDataset(Dataset):

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Dataloader specific arguments')
        parser.add_argument('-input_img', default='data/0.9/data_img.h5',
                                help='HDF5 file with image features')
        parser.add_argument('-input_ques', default='data/0.9/visdial_data.h5',
                                help='HDF5 file with preprocessed questions')
        parser.add_argument('-input_json', default='data/0.9/visdial_params.json',
                                help='JSON file with image paths and vocab')
        parser.add_argument('-sim_file', default='data/0.9/sim_data.h5',
                                help='HDF5 file with similaritiy scores')
 
        parser.add_argument('-img_norm', default=1, choices=[1, 0],
                                help='normalize the image feature. 1=yes, 0=no')
        parser.add_argument('-size_limit', type=int, default=-1, help='maximum size of dataset')
        parser.add_argument('-breakdown_analysis', action='store_true',
                                help='break down statistics by question type')
        parser.add_argument('-similarities', action='store_true',
                                help='compute similarities for options to ground truth')
        return parser

    def __init__(self, args, subsets):
        """Initialize the dataset with splits given by 'subsets', where
        subsets is taken from ['train', 'val', 'test']
        """
        super().__init__()
        self.args = args
        self.subsets = tuple(subsets)

        print("Dataloader loading json file: {}".format(args.input_json))
        with open(args.input_json, 'r') as info_file:
            info = json.load(info_file)
            # possible keys: {'ind2word', 'word2ind', 'unique_img_(split)'}
            for key, value in iteritems(info):
                setattr(self, key, value)

        # add <START> and <END> to vocabulary
        word_count = len(self.word2ind)
        self.word2ind['<START>'] = word_count + 1
        self.word2ind['<END>'] = word_count + 2
        self.start_token = self.word2ind['<START>']
        self.end_token = self.word2ind['<END>']

        # padding + <START> + <END> token
        self.vocab_size = word_count + 3
        print("Vocab size with <START>, <END>: {}".format(self.vocab_size))

        # construct reverse of word2ind after adding tokens
        self.ind2word = {
            int(ind): word
            for word, ind in iteritems(self.word2ind)
        }

        print("Dataloader loading h5 file: {}".format(args.input_ques))
        ques_file = h5py.File(args.input_ques, 'r')

        print("Dataloader loading h5 file: {}".format(args.input_img))
        img_file = h5py.File(args.input_img, 'r')

        # load all data mats from ques_file into this
        self.data = {}

        # map from load to save labels
        io_map = {
            'ques_{}': '{}_ques',
            'ques_length_{}': '{}_ques_len',
            'ans_{}': '{}_ans',
            'ans_length_{}': '{}_ans_len',
            'img_pos_{}': '{}_img_pos',
            'cap_{}': '{}_cap',
            'cap_length_{}': '{}_cap_len',
            'opt_{}': '{}_opt',
            'opt_length_{}': '{}_opt_len',
            'opt_list_{}': '{}_opt_list',
            'num_rounds_{}': '{}_num_rounds',
            'ans_index_{}': '{}_ans_ind'
        }

        # processing every split in subsets
        # We will redefine the subsets be:
        # 'train': 70000 sampled examples from v1.0
        # 'val': 12783 examples from v1.0
        # 'test' : actual 'val'
        if len(subsets) > 1:
            print('Not supported yet')
            return
        split_map = { 'train': 'train', 'val': 'train', 'test': 'val' }
        for dtype in subsets:  # dtype is in ['train', 'val', 'test']
            underlying_dtype = split_map[dtype]
            print("\nProcessing split [{}]...".format(dtype))
            # read the question, answer, option related information
            for load_label, save_label in iteritems(io_map):
                if load_label.format(underlying_dtype) not in ques_file:
                    continue
                self.data[save_label.format(dtype)] = torch.from_numpy(
                    np.array(ques_file[load_label.format(underlying_dtype)], dtype='int64'))

            print("Reading image features...")
            img_feats = torch.from_numpy(np.array(img_file['images_' + underlying_dtype]))

            if args.img_norm:
                print("Normalizing image features...")
                img_feats = F.normalize(img_feats, dim=1, p=2)

            # save image features
            self.data[dtype + '_img_fv'] = img_feats
            img_fnames = getattr(self, 'unique_img_' + underlying_dtype)
            self.data[dtype + '_img_fnames'] = img_fnames
            #print(img_fnames)

            # record some stats, will be transferred to encoder/decoder later
            # assume similar stats across multiple data subsets
            # maximum number of questions per image, ideally 10
            self.max_ques_count = self.data[dtype + '_ques'].size(1)
            # maximum length of question
            self.max_ques_len = self.data[dtype + '_ques'].size(2)
            # maximum length of answer
            self.max_ans_len = self.data[dtype + '_ans'].size(2)

        ignore = ['opt_list', 'opt_len']
        for key in self.data:
            ignore_key = False
            for ignore_string in ignore:
                if ignore_string in key:
                    ignore_key = True
                    break
            if not ignore_key:
                if dtype == 'train':
                    self.data[key] = self.data[key][:70000]
                if dtype == 'val':
                    self.data[key] = self.data[key][80000:]
                if dtype == 'test':
                    self.data[key] = self.data[key][:3000]

        # reduce amount of data for preprocessing in fast mode
        if args.overfit:
            self.data[dtype + '_img_fv'] = self.data[dtype + '_img_fv'][:5]
            self.data[dtype + '_img_fnames'] = self.data[dtype + '_img_fnames'][:5]
        if args.size_limit > 0:
            size_limit = min(args.size_limit, list(self.data[dtype + '_img_fv'].shape)[0])
            self.data[dtype + '_img_fv'] = self.data[dtype + '_img_fv'][:size_limit]
            self.data[dtype + '_img_fnames'] = self.data[dtype + '_img_fnames'][:size_limit]

        self.num_data_points = {}
        for dtype in subsets:
            self.num_data_points[dtype] = len(self.data[dtype + '_img_fv'])
            print("[{0}] no. of threads: {1}".format(dtype, self.num_data_points[dtype]))
        print("\tMax no. of rounds: {}".format(self.max_ques_count))
        print("\tMax ques len: {}".format(self.max_ques_len))
        print("\tMax ans len: {}".format(self.max_ans_len))
	
        # prepare history
        for dtype in subsets:
            self._process_history(dtype)

            # 1 indexed to 0 indexed
            self.data[dtype + '_opt'] -= 1
            if dtype + '_ans_ind' in self.data:
                self.data[dtype + '_ans_ind'] -= 1

        if args.breakdown_analysis:
            # in val_ques first index is rounds, second index is question, third index is words
            # print(self.data['val_ques'].size(0))
            # print(self.data['val_ans'].size())
            self.data[dtype +'_type'] = collections.defaultdict(dict) 
            for i in range(self.data[dtype + '_ques'].size(0)):
                for j in range(self.data[dtype + '_num_rounds'][i]):
                    is_color = False
                    is_yesno = False
                    is_count = False
                    question = ""
                    for k in range(self.data[dtype + '_ques_len'][i][j]):
                        word = self.ind2word[self.data[dtype + '_ques'][i][j][k].item()]
                        question = question + " " + word
                        if (word == "color"):
                            is_color = True
                    answer = ""
                    for k in range(self.data[dtype + '_ans_len'][i][j]):
                        word = self.ind2word[self.data[dtype + '_ans'][i][j][k].item()]
                        answer = answer + " " + word
                        if (word == "yes" or word == "no"):
                            is_yesno = True
                    if (any(char.isdigit() for char in answer)):
                        is_count = True
                    #if i < 20:
                    #    print(question)
                    #    print(answer)
                    #    print(is_yesno)
                    if (is_yesno):
                        self.data[dtype + '_type'][i][j] = "yn"
                    elif (is_color):
                        self.data[dtype + '_type'][i][j] = "color"
                    elif (is_count):
                        self.data[dtype + '_type'][i][j] = "count"
                    else: 
                        self.data[dtype +'_type'][i][j] = "other"
        if args.similarities:
            num_data_points = self.num_data_points['train']
            # n * 10 * 100
            self.data['train_words'] = [None] * num_data_points

            nlp = spacy.load('en_core_web_lg', disable=['parser','tagger','ner'])
            print ("REACHES HERE")
            self.data['train_sim'] = [None] * num_data_points
            
            opt_size = self.data['train_opt_list'].size(0)
            nlp_list = [None] * opt_size
            option_list = []
            for i in range(opt_size):
                converted_string = convert_to_string(self.data['train_opt_list'][i],self.ind2word)
                nlp_list[i] = converted_string
            for doc in nlp.pipe(nlp_list):
                option_list.append(doc)

            ans_list = []
            for i in range(num_data_points):
                for j in range(self.data['train_num_rounds'][i]):
                    converted_string = convert_to_string(self.data['train_ans'][i][j], self.ind2word)
                    ans_list.append(converted_string)
            ans_nlp_list = []
            for doc in nlp.pipe(ans_list):
                ans_nlp_list.append(doc)
    
            print("REACHES AFTER NLP")

            ans_index = 0
            for i in range(num_data_points):
                self.data['train_sim'][i] = [None] * self.data['train_num_rounds'][i].item()
                self.data['train_words'][i] = [None] * self.data['train_num_rounds'][i].item()
                for j in range(self.data['train_num_rounds'][i]):
                    self.data['train_sim'][i][j] = [None] * self.data['train_opt'][i][j].size(0)
                    self.data['train_words'][i][j] = [None] * self.data['train_opt'][i][j].size(0)
                    #answer = convert_to_string(self.data['train_ans'][i][j], self.ind2word)
                    #answer_nlp = nlp(answer)
                    answer_nlp = ans_nlp_list[ans_index]
                    #print(self.data['train_opt_list'].size())
                    for k in range(self.data['train_opt'][i][j].size(0)):
                        tens_ind = self.data['train_opt'][i][j][k].item()
                        #option = convert_to_string(self.data['train_opt_list'][tens_ind], self.ind2word)
                        option_nlp = option_list[tens_ind]
                        similarity = answer_nlp.similarity(option_nlp)
                        # self.data['train_words'][i][j][k] = option_nlp.text
                        #only applies to spacy
                        if (similarity < 0):
                            similarity = 0
                        self.data['train_sim'][i][j][k] = similarity
                        #print (self.data['train_opt'][i][j][k])
                                                
                    ans_index = ans_index + 1    
                    '''
                    for k in range(self.data['train_ans_len'][i][j]):
                        word = self.ind2word[self.data['train_ans'][i][j][k].item()]
                        self.data['train_sim'][i] = [1,2,3]
                    '''
            
            print("REACHES AFTER")
            """
            self.data['train_sim'] = np.array(self.data['train_sim']) 
            sim_file = h5py.File(args.sim_file, 'w')
            sim_file.create_dataset('sim', data = self.data['train_sim'])
            sim_file.close()
            """
            # array_train_words = self.data['train_words']
            # with open('data/0.9/words.json', 'w') as outfile:
            #     json.dump({ 'data': array_train_words }, outfile)
            
        sim_file = h5py.File(args.sim_file, 'r')
        self.data['train_sim'] = sim_file.get('sim')
        print("READ FILE")
        #generating probabilities
        theta = 20
        sim_scores = self.data['train_sim']
        print(sim_scores.shape)
        order = np.argsort(sim_scores, axis=2)
        print(order.shape)
        order = np.flip(order, 2)
        weights = [0.5, 0.2, 0.1, 0.1, 0.1]
        sim_scores_normal = np.zeros(sim_scores.shape)
        for i in range(5):
            top_order_i = order[:,:,i]
            for j in range(sim_scores_normal.shape[0]):
                for k in range(sim_scores_normal.shape[1]):
                    sim_scores_normal[j][k][top_order_i[j][k]] = weights[i]
        print(sim_scores_normal[0][0])
        self.data['train_sim'] = sim_scores_normal
        self.data['train_sim'] = torch.tensor(self.data['train_sim']).type(torch.FloatTensor)
        print("size in dataloader")
        print(self.data['train_sim'].size())

        # default pytorch loader dtype is set to train
        if 'train' in subsets:
            self._split = 'train'
        else:
            self._split = subsets[0]

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets  # ['train', 'val', 'test']
        self._split = split

    # ------------------------------------------------------------------------
    # methods to override - __len__ and __getitem__ methods
    # ------------------------------------------------------------------------

    def __len__(self):
        return self.num_data_points[self._split]

    def __getitem__(self, idx):
        dtype = self._split
        item = {'index': idx}
        item['num_rounds'] = self.data[dtype + '_num_rounds'][idx]

        # get image features
        item['img_feat'] = self.data[dtype + '_img_fv'][idx]
        item['img_fnames'] = self.data[dtype + '_img_fnames'][idx]

        # get question tokens
        item['ques'] = self.data[dtype + '_ques'][idx]
        item['ques_len'] = self.data[dtype + '_ques_len'][idx]

        # get history tokens
        item['hist_len'] = self.data[dtype + '_hist_len'][idx]
        item['hist'] = self.data[dtype + '_hist'][idx]

        # get type tokens
        item['type'] =  self.data[dtype + '_type'][idx] if (dtype + '_type') in self.data else ''

        #get sim tokens
        item['sim'] = self.data[dtype + '_sim'][idx] if (dtype + '_sim') in self.data else torch.zeros([100], dtype=torch.float32)
        

        # get options tokens
        opt_inds = self.data[dtype + '_opt'][idx]
        opt_size = list(opt_inds.size())
        new_size = torch.Size(opt_size + [-1])
        ind_vector = opt_inds.view(-1)

        option_in = self.data[dtype + '_opt_list'].index_select(0, ind_vector)
        option_in = option_in.view(new_size)

        opt_len = self.data[dtype + '_opt_len'].index_select(0, ind_vector)
        opt_len = opt_len.view(opt_size)

        item['opt'] = option_in
        item['opt_len'] = opt_len
        #if dtype != 'test':
        ans_ind = self.data[dtype + '_ans_ind'][idx]
        item['ans_ind'] = ans_ind.view(-1)

        # convert zero length sequences to one length
        # this is for handling empty rounds of v1.0 test, they will be dropped anyway
        if dtype == 'test':
            item['ques_len'][item['ques_len'] == 0] += 1
            item['opt_len'][item['opt_len'] == 0] += 1
            item['hist_len'][item['hist_len'] == 0] += 1
        return item

    #-------------------------------------------------------------------------
    # collate function utilized by dataloader for batching
    #-------------------------------------------------------------------------

    def collate_fn(self, batch):
        dtype = self._split
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        out = {}
        for key in merged_batch:
            if key in {'index', 'num_rounds', 'img_fnames', 'type'}:
                out[key] = merged_batch[key]
            elif key in {'cap_len'}:
                out[key] = torch.Tensor(merged_batch[key]).long()
            else:
                out[key] = torch.stack(merged_batch[key], 0)

        # Dynamic shaping of padded batch
        out['hist'] = out['hist'][:, :, :torch.max(out['hist_len'])].contiguous()
        out['ques'] = out['ques'][:, :, :torch.max(out['ques_len'])].contiguous()
        out['opt'] = out['opt'][:, :, :, :torch.max(out['opt_len'])].contiguous()
        #out['sim'] = out['sim'][:, :, :, :torch.max(out['sim'])].contiguous()

        batch_keys = ['img_fnames', 'num_rounds', 'img_feat', 'hist',
                      'hist_len', 'ques', 'ques_len', 'opt', 'opt_len','type', 'sim']
        #if dtype != 'test':
        batch_keys.append('ans_ind')
        return {key: out[key] for key in batch_keys}

    #-------------------------------------------------------------------------
    # preprocessing functions
    #-------------------------------------------------------------------------

    def _process_history(self, dtype):
        """Process caption as well as history. Optionally, concatenate history
        for lf-encoder."""
        captions = self.data[dtype + '_cap']
        questions = self.data[dtype + '_ques']
        ques_len = self.data[dtype + '_ques_len']
        cap_len = self.data[dtype + '_cap_len']
        max_ques_len = questions.size(2)

        answers = self.data[dtype + '_ans']
        ans_len = self.data[dtype + '_ans_len']
        num_convs, num_rounds, max_ans_len = answers.size()

        if self.args.concat_history:
            self.max_hist_len = min(num_rounds * (max_ques_len + max_ans_len), 300)
            history = torch.zeros(num_convs, num_rounds, self.max_hist_len).long()
        elif self.args.partial_concat_history:
            self.max_hist_len = min(3 * (max_ques_len + max_ans_len), 300)
            history = torch.zeros(num_convs, num_rounds, self.max_hist_len).long()
        else:
            history = torch.zeros(num_convs, num_rounds, max_ques_len + max_ans_len).long()
        hist_len = torch.zeros(num_convs, num_rounds).long()

        # go over each question and append it with answer
        for th_id in range(num_convs):
            clen = cap_len[th_id]
            hlen = min(clen, max_ques_len + max_ans_len)
            for round_id in range(num_rounds):
                if round_id == 0:
                    # first round has caption as history
                    history[th_id][round_id][:max_ques_len + max_ans_len] \
                        = captions[th_id][:max_ques_len + max_ans_len]
                else:
                    qlen = ques_len[th_id][round_id - 1]
                    alen = ans_len[th_id][round_id - 1]
                    # if concat_history, string together all previous question-answer pairs
                    if self.args.concat_history:
                        history[th_id][round_id][:hlen] = history[th_id][round_id - 1][:hlen]
                        history[th_id][round_id][hlen] = self.word2ind['<END>']
                        if qlen > 0:
                            history[th_id][round_id][hlen + 1:hlen + qlen + 1] \
                                = questions[th_id][round_id - 1][:qlen]
                        if alen > 0:
                            # print(round_id, history[th_id][round_id][:10], answers[th_id][round_id][:10])
                            history[th_id][round_id][hlen + qlen + 1:hlen + qlen + alen + 1] \
                                = answers[th_id][round_id - 1][:alen]
                        hlen = hlen + qlen + alen + 1
                    elif self.args.partial_concat_history:
                        clen = max_ques_len + max_ans_len
                        history[th_id][round_id][:clen] \
                            = captions[th_id][:clen]
                        history[th_id][round_id][clen] = self.word2ind['<END>']
                        if qlen > 0:
                            history[th_id][round_id][clen + 1:clen + 1 + qlen] = questions[th_id][round_id - 1][:qlen]
                        if alen > 0:
                            history[th_id][round_id][clen + 1 + qlen:clen + 1 + qlen + alen] \
                                = answers[th_id][round_id - 1][:alen]
                        hlen = clen + 1 + alen + qlen
                    # else, history is just previous round question-answer pair
                    else:
                        if qlen > 0:
                            history[th_id][round_id][:qlen] = questions[th_id][round_id - 1][:qlen]
                        if alen > 0:
                            history[th_id][round_id][qlen:qlen + alen] \
                                = answers[th_id][round_id - 1][:alen]
                        hlen = alen + qlen
                # save the history length
                hist_len[th_id][round_id] = hlen

        self.data[dtype + '_hist'] = history
        self.data[dtype + '_hist_len'] = hist_len
