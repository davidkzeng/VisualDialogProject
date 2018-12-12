import os
import json
import spacy
from six import iteritems
import collections

import h5py
import numpy as np
from matplotlib import pyplot
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import convert_to_string

args = {}
args['sim_file'] = 'data/0.9/sim_data.h5'
sim_file = h5py.File(args['sim_file'], 'r')
data = sim_file.get('sim')
data_cutoff = data[:10]

"""
data_cutoff = data_cutoff.flatten()
pyplot.hist(data_cutoff, bins=20, density=True)
pyplot.title("Distribution of Cosine Similarity to Ground Truth")
pyplot.ylabel("Frequency")
pyplot.xlabel("Similarity")
pyplot.show()
pyplot.savefig('sim_dist.png')
"""
words = None
with open('data/0.9/words.json') as json_data:
    words = json.load(json_data)['data']

theta = 20
sim_scores = data_cutoff
sim_scores = np.multiply(theta, sim_scores)
sim_scores = np.exp(sim_scores)
sim_score_totals = np.sum(sim_scores, axis=2)
sim_scores_normal = sim_scores / sim_score_totals[:, :, np.newaxis]

for num in range(10):
    for i in range(10):
        first = sim_scores_normal[num][i]
        orig = data_cutoff[num][i]
        first_words = words[num][i]
            
        p = first.argsort()[::-1]
        first_words = np.array(first_words)
        first_words = first_words[p]
        print(first_words[:10])
        first.sort()
        orig.sort()
        print(orig[::-1])
        pyplot.bar(np.array([i for i in range(100)]), first)
        pyplot.show()
