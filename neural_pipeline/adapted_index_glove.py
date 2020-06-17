import dill
import os
import re
import shutil
import random
import torch
import argparse
import time
from collections import Counter, defaultdict
from adapted_data_to_input import Indexer
import numpy as np
import os
import bcolz
import gensim

def get_tokens(train_paths, word2vec_model, word2vec_map, return_freq_cut_set):
    """Count tokens. `data_group` has pickle files: (query, graph_mix, target_graph_id)."""
    token2count = Counter()
    i = 0
    stmt_labels = set()

    start = time.time()

    for path in train_paths:
        train_list = [os.path.join(path, file_name) for file_name in os.listdir(path)]

        for file_name in train_list:
            _, _, graph_mix, _ = dill.load(open(file_name, "rb"))
            for ere in graph_mix.eres.values():
                for label in [item for item in ere.label if item != 'Relation']:
                    try:
                        word2vec_model.get_vector(('_').join(label.split()))
                    except KeyError:
                        if ('_').join([item.lower() for item in label.split()]) not in word2vec_map.keys():
                            for word in label.split():
                                try:
                                    word2vec_model.get_vector(word)
                                except KeyError:
                                    if word.lower() not in word2vec_map.keys():
                                        token2count[word] += 1
                                    else:
                                        token2count[word.lower()] += 1
                                else:
                                    token2count[word] += 1
                        else:
                            token2count[('_').join([item.lower() for item in label.split()])] += 1
                    else:
                        token2count[('_').join(label.split())] += 1

            for stmt in graph_mix.stmts.values():
                for label in stmt.label:
                    [stmt_labels.add(word.lower()) for word in re.findall('[a-zA-Z][^A-Z]*', label)]

            if i % 10000 == 0:
                print("... processed %d files (%.2fs)." % (i, time.time() - start))
                start = time.time()
            i += 1

    return set(list(zip(*token2count.most_common(return_freq_cut_set)))[0]), stmt_labels


def create_indexers_for_corpus(train_paths, indexer_dir, word2vec_model, word2vec_map, emb_dim, return_freq_cut_set):
    ere_indexer = Indexer()
    ere_indexer.word_to_index = {}
    ere_indexer.index_to_word = {}
    ere_indexer.size = 0

    stmt_indexer = Indexer()
    stmt_indexer.word_to_index = {}
    stmt_indexer.index_to_word = {}
    stmt_indexer.size = 0

    token_set, stmt_labels = get_tokens(train_paths, word2vec_model, word2vec_map, return_freq_cut_set)

    min_vec = np.full(emb_dim, np.inf, dtype=np.float64)
    max_vec = -1 * np.full(emb_dim, np.inf, dtype=np.float64)

    for item in word2vec_model.vocab.keys():
        min_vec = np.amin(np.concatenate([min_vec.reshape((1, -1)), word2vec_model.get_vector(item).reshape((1, -1))], axis=0), axis=0)
        max_vec = np.amax(np.concatenate([max_vec.reshape((1, -1)), word2vec_model.get_vector(item).reshape((1, -1))], axis=0), axis=0)

    word2vec_tokens = set.intersection(token_set, set.union(set(word2vec_model.vocab.keys()), set(word2vec_map.keys())))
    print(len(word2vec_tokens))

    [ere_indexer.get_index(item, add=True) for item in word2vec_tokens]
    num_word2vec_ere = len(word2vec_tokens)

    ere_indexer.get_index('<CAT>Event', add=True)
    ere_indexer.get_index('<CAT>Relation', add=True)
    ere_indexer.get_index('<CAT>Entity', add=True)
    [ere_indexer.get_index(item, add=True) for item in token_set - word2vec_tokens]

    ere_emb_mat = np.zeros((ere_indexer.size, emb_dim), dtype=np.float64)

    for iter in range(num_word2vec_ere):
        word = ere_indexer.get_word(iter)

        if word not in word2vec_map.keys():
            ere_emb_mat[iter] = word2vec_model.get_vector(word)
        else:
            temp = None
            for item in word2vec_map[word]:
                if item.lower == item:
                    temp = item
                    break

            if temp:
                ere_emb_mat[iter] = word2vec_model.get_vector(temp)
            else:
                ere_emb_mat[iter] = word2vec_model.get_vector(word2vec_map[word][0])

    for iter in range(num_word2vec_ere, ere_indexer.size):
        ere_emb_mat[iter] = np.random.uniform(min_vec, max_vec, emb_dim)

    word2vec_tokens = set.intersection(stmt_labels, set.union(set(word2vec_model.vocab.keys()), set(word2vec_map.keys())))

    [stmt_indexer.get_index(item, add=True) for item in word2vec_tokens]
    num_word2vec_stmt = len(word2vec_tokens)

    stmt_indexer.get_index('<Typing>No', add=True)
    stmt_indexer.get_index('<Typing>Yes', add=True)
    [stmt_indexer.get_index(item, add=True) for item in stmt_labels - word2vec_tokens]

    stmt_emb_mat = np.zeros((stmt_indexer.size, emb_dim), dtype=np.float64)

    for iter in range(num_word2vec_stmt):
        word = stmt_indexer.get_word(iter)

        if word not in word2vec_map.keys():
            stmt_emb_mat[iter] = word2vec_model.get_vector(word)
        else:
            temp = None
            for item in word2vec_map[word]:
                if item.lower == item:
                    temp = item
                    break

            if temp:
                stmt_emb_mat[iter] = word2vec_model.get_vector(temp)
            else:
                stmt_emb_mat[iter] = word2vec_model.get_vector(word2vec_map[word][0])

    for iter in range(num_word2vec_stmt, stmt_indexer.size):
        stmt_emb_mat[iter] = np.random.uniform(min_vec, max_vec, emb_dim)

    dill.dump((ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt), open(os.path.join(indexer_dir, "indexers.p"), "wb"))

    return (ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt)


def convert_labels_to_indices(graph_mix, indexer_info):
    ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt = indexer_info

    ere_mat_ind = Indexer()
    stmt_mat_ind = Indexer()

    ere_mat_ind.word_to_index = {}
    ere_mat_ind.index_to_word = {}
    ere_mat_ind.size = 0

    stmt_mat_ind.word_to_index = {}
    stmt_mat_ind.index_to_word = {}
    stmt_mat_ind.size = 0

    for ere_id in graph_mix.eres.keys():
        ere_mat_ind.get_index(ere_id, add=True)

    for stmt_id in graph_mix.stmts.keys():
        stmt_mat_ind.get_index(stmt_id, add=True)

    adj_ere_to_stmt_head = torch.zeros((ere_mat_ind.size, stmt_mat_ind.size), dtype=torch.uint8)
    adj_ere_to_stmt_tail = torch.zeros((ere_mat_ind.size, stmt_mat_ind.size), dtype=torch.uint8)
    adj_ere_to_type_stmt = torch.zeros((ere_mat_ind.size, stmt_mat_ind.size), dtype=torch.uint8)
    #inv_sqrt_degree_mat_eres = torch.zeros((ere_mat_ind.size, ere_mat_ind.size))
    #inv_sqrt_degree_mat_stmts = torch.zeros((stmt_mat_ind.size, stmt_mat_ind.size))
    ere_labels = [[] for _ in range(ere_mat_ind.size)]
    stmt_labels = [[] for _ in range(stmt_mat_ind.size)]

    for ere_iter in range(ere_mat_ind.size):
        ere_id = ere_mat_ind.get_word(ere_iter)
        head_stmt_keys = [stmt_mat_ind.get_index(stmt_key, add=False) for stmt_key in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[stmt_key].head_id == ere_id and graph_mix.stmts[stmt_key].tail_id]
        type_stmt_keys = [stmt_mat_ind.get_index(stmt_key, add=False) for stmt_key in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[stmt_key].head_id == ere_id and not graph_mix.stmts[stmt_key].tail_id]
        tail_stmt_keys = [stmt_mat_ind.get_index(stmt_key, add=False) for stmt_key in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[stmt_key].tail_id == ere_id]

        adj_ere_to_stmt_head[ere_iter][head_stmt_keys] = 1
        adj_ere_to_stmt_tail[ere_iter][tail_stmt_keys] = 1
        adj_ere_to_type_stmt[ere_iter][type_stmt_keys] = 1
        #inv_sqrt_degree_mat_eres[ere_iter][ere_iter] = 1 / np.sqrt(len(graph_mix.eres[ere_id].stmt_ids) + 1)
        ere_labels[ere_iter].append([ere_indexer.get_index('<CAT>' + graph_mix.eres[ere_id].category, add=False)])

        for label in [item for item in graph_mix.eres[ere_id].label if item != 'Relation']:
            temp = []
            if ('_').join(label.split()) in ere_indexer.word_to_index.keys():
                temp.append(ere_indexer.get_index(('_').join(label.split()), add=False))
            elif ('_').join([word.lower() for word in label.split()]) in ere_indexer.word_to_index.keys():
                temp.append(ere_indexer.get_index(('_').join([word.lower() for word in label.split()]), add=False))
            else:
                for word in label.split():
                    if word in ere_indexer.word_to_index.keys():
                        temp.append(ere_indexer.get_index(word, add=False))
                    elif word.lower() in ere_indexer.word_to_index.keys():
                        temp.append(ere_indexer.get_index(word.lower(), add=False))

            if len(temp) > 0:
                ere_labels[ere_iter].append(temp)

    for stmt_iter in range(stmt_mat_ind.size):
        stmt_id = stmt_mat_ind.get_word(stmt_iter)

        stmt_labels[stmt_iter].append([stmt_indexer.get_index('<Typing>' + ('No' if graph_mix.stmts[stmt_id].tail_id else 'Yes'), add=False)])
        #inv_sqrt_degree_mat_stmts[stmt_iter][stmt_iter] = 1 / np.sqrt(3) if graph_mix.stmts[stmt_id].tail_id else 1 / np.sqrt(2)

        for label in graph_mix.stmts[stmt_id].label:
            temp = []
            for word in re.findall('[a-zA-Z][^A-Z]*', label):
                if word.lower() in stmt_indexer.word_to_index.keys():
                    temp.append(stmt_indexer.get_index(word.lower(), add=False))

            stmt_labels[stmt_iter].append(temp)

    return {'graph_mix': graph_mix, 'ere_mat_ind': ere_mat_ind, 'stmt_mat_ind': stmt_mat_ind, 'adj_head': adj_ere_to_stmt_head, 'adj_tail': adj_ere_to_stmt_tail, 'adj_type': adj_ere_to_type_stmt, 'ere_labels': ere_labels, 'stmt_labels': stmt_labels, 'num_word2vec_ere': num_word2vec_ere, 'num_word2vec_stmt': num_word2vec_stmt}


def index(data_dir, indexer_dir, emb_dim, return_freq_cut_set):
    indexed_data_dir = data_dir + '_Indexed'
    indexer_dir = os.path.join(indexed_data_dir, indexer_dir)

    if not os.path.exists(indexed_data_dir):
        os.makedirs(indexed_data_dir)

    if not os.path.exists(indexer_dir):
        os.makedirs(indexer_dir)

    data_paths = [os.path.join(data_dir, item) for item in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, item))]
    train_paths = [os.path.join(item, 'Train') for item in data_paths]
    val_paths = [os.path.join(item, 'Val') for item in data_paths]
    test_paths = [os.path.join(item, 'Test') for item in data_paths]

    indexed_data_paths = [os.path.join(indexed_data_dir, item + '_Indexed') for item in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, item))]

    [os.makedirs(path) for path in indexed_data_paths if not os.path.exists(path)]
    [os.makedirs(os.path.join(path, data_cat)) if not os.path.exists(os.path.join(path, data_cat)) else None for data_cat in ['Train', 'Val', 'Test'] for path in indexed_data_paths]

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/home/atomko/Google_News_Embeds/GoogleNews-vectors-negative300.bin', binary=True)
    word2vec_vocab = word2vec_model.vocab.keys()

    word2vec_map = defaultdict(list)

    for item in word2vec_vocab:
        word2vec_map[item.lower()].append(item)

    indexer_info = create_indexers_for_corpus(train_paths, indexer_dir, word2vec_model, word2vec_map, emb_dim, return_freq_cut_set)

    for data_iter, paths in enumerate([train_paths, val_paths, test_paths]):
        for path_iter, path in enumerate(paths):
            for file_name in os.listdir(path):
                _, query, graph_mix, target_graph_id = dill.load(open(os.path.join(path, file_name), 'rb'))

                graph_dict = convert_labels_to_indices(graph_mix, indexer_info)

                query_stmt_indices = [graph_dict['stmt_mat_ind'].get_index(stmt_id, add=False) for stmt_id in query]

                query_ere_indices = set.union(*[set([graph_dict['ere_mat_ind'].get_index(graph_dict['graph_mix'].stmts[stmt].head_id, add=False), graph_dict['ere_mat_ind'].get_index(graph_dict['graph_mix'].stmts[stmt].tail_id, add=False)])
                                                if graph_dict['graph_mix'].stmts[stmt].tail_id else set([graph_dict['ere_mat_ind'].get_index(graph_dict['graph_mix'].stmts[stmt].head_id, add=False)]) for stmt in query])

                graph_dict['query_stmts'] = query_stmt_indices
                graph_dict['query_eres'] = query_ere_indices
                graph_dict['target_graph_id'] = target_graph_id

                if data_iter == 0:
                    dill.dump(graph_dict, open(os.path.join(indexed_data_paths[path_iter], 'Train', file_name), 'wb'))
                elif data_iter == 1:
                    dill.dump(graph_dict, open(os.path.join(indexed_data_paths[path_iter], 'Val', file_name), 'wb'))
                else:
                    dill.dump(graph_dict, open(os.path.join(indexed_data_paths[path_iter], 'Test', file_name), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/home/atomko/backup_drive/AIDA_Data_Gen_10_6/Mixtures_New', help='Directory containing folders of different mixture types')
    parser.add_argument("--indexer_dir", type=str, default='Indexers', help='Directory where the indexers will be written')
    parser.add_argument("--emb_dim", type=int, default=300, help='Index the x most frequent ERE label tokens')
    parser.add_argument("--return_freq_cut_set", type=int, default=50000, help='Index the x most frequent ERE label tokens')

    args = parser.parse_args()
    locals().update(vars(args))

    index(data_dir, indexer_dir, emb_dim, return_freq_cut_set)

