import numpy as np
import io
import os
import re
import json
import dill
import torch
import h5py
import re
from collections import defaultdict
from copy import deepcopy
from adapted_data_to_input import Indexer
#import bcolz
#import gensim

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder

mix_list = os.listdir('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New_70k/Debug_Mixtures_New')

elmo = ElmoEmbedder(weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
                    options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json',
                    cuda_device=0)

def get_tokens(train_dir):
    train_paths = [os.path.join(train_dir, item) for item in os.listdir(train_dir)]

    tokens = set()

    for file_iter, file_name in enumerate(train_paths):
        if 'self' in file_name:
            graph_mix, _, _, _, _ = dill.load(open(file_name, 'rb'))
        else:
            graph_mix, _, _, _, target_event_stmt = dill.load(open(file_name, 'rb'))

        for stmt in graph_mix.stmts.values():
            for label in re.sub("[._]", " ", stmt.raw_label).split():
                [tokens.add(word.lower()) for word in re.findall('[a-zA-Z][^A-Z]*', label)]

    return tokens

tokens = list(get_tokens('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New_70k/Debug_Mixtures_New/Train'))
print(tokens)
print(len(tokens))

ont_tok_indexer = Indexer()
ont_tok_indexer.word_to_index = {}
ont_tok_indexer.index_to_word = {}
ont_tok_indexer.size = 0

for tok in tokens:
    ont_tok_indexer.get_index(tok, add=True)

ont_tok_indexer.get_index('<Typing>No', add=True)
ont_tok_indexer.get_index('<Typing>Yes', add=True)

print(ont_tok_indexer.size)

ont_tok_emb = np.zeros((ont_tok_indexer.size, 1024))

for iter in range(len(tokens)):
    ont_tok_emb[iter] = elmo.embed_sentence([tokens[iter]])[0].squeeze(0)

min_vec = np.full(1024, np.inf, dtype=np.float64)
max_vec = -1 * np.full(1024, np.inf, dtype=np.float64)

for iter in range(len(tokens)):
    min_vec = np.amin(np.concatenate([min_vec.reshape((1, -1)), ont_tok_emb[iter].reshape((1, -1))], axis=0), axis=0)
    max_vec = np.amax(np.concatenate([max_vec.reshape((1, -1)), ont_tok_emb[iter].reshape((1, -1))], axis=0), axis=0)

for iter in range(len(tokens), ont_tok_indexer.size):
    ont_tok_emb[iter] = np.random.uniform(min_vec, max_vec, 1024)

dill.dump((ont_tok_emb, ont_tok_indexer), open('/home/atomko/backup_drive/Summer_2020/ont_embs_for_salad.p', 'wb'))