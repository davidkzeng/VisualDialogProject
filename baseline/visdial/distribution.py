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
data_cutoff = data[:100]
data_cutoff = data_cutoff.flatten()
print(data.shape)

pyplot.hist(data_cutoff, bins=20)
pyplot.show()
