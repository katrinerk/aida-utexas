# Original author: Su Wang, 2019
# Modified by Alex Tomkovich in 2019/2020

######
# This file converts names (e.g., "Winston Churchill") and labels (e.g., "person")
# into indices to be referenced in an embeddings matrix.
######

import dill
import re
import argparse
import time
from collections import Counter
import numpy as np
import os
import gensim
from gen_single_doc_graphs import Indexer, verify_dir

# Count the number of times each token appears in the training set in names or labels
def get_tokens(train_path, word2vec_model, return_freq_cut_set):
    token2count = Counter()
    i = 0
    stmt_labels = set()

    start = time.time()

    train_list = [os.path.join(train_path, file_name) for file_name in os.listdir(train_path)]

    for file_name in train_list:
        _, _, graph_mix, _ = dill.load(open(file_name, "rb"))
        for ere in graph_mix.eres.values():
            for label in [item for item in ere.label if item != 'Relation']:
                try:
                    word2vec_model.get_vector(('_').join(label.split()))
                except KeyError:
                    for word in label.split():
                        token2count[word] += 1
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

# Create indexers based on vocabulary in training graph salads.
def create_indexers_for_corpus(train_paths, indexer_dir, word2vec_model, emb_dim, return_freq_cut_set):
    ere_indexer = Indexer()
    ere_indexer.word_to_index = {}
    ere_indexer.index_to_word = {}
    ere_indexer.size = 0

    stmt_indexer = Indexer()
    stmt_indexer.word_to_index = {}
    stmt_indexer.index_to_word = {}
    stmt_indexer.size = 0

    # Get set of tokens to index
    token_set, stmt_labels = get_tokens(train_paths, word2vec_model, return_freq_cut_set)

    # Determine space of representations for Word2Vec embeddings
    min_vec = np.full(emb_dim, np.inf, dtype=np.float64)
    max_vec = -1 * np.full(emb_dim, np.inf, dtype=np.float64)

    for item in word2vec_model.vocab.keys():
        min_vec = np.amin(np.concatenate([min_vec.reshape((1, -1)), word2vec_model.get_vector(item).reshape((1, -1))], axis=0), axis=0)
        max_vec = np.amax(np.concatenate([max_vec.reshape((1, -1)), word2vec_model.get_vector(item).reshape((1, -1))], axis=0), axis=0)

    # Determine which observed tokens are already in the Word2Vec vocabulary; index these first
    word2vec_tokens = set.intersection(token_set, set(word2vec_model.vocab.keys()))

    [ere_indexer.get_index(item, add=True) for item in word2vec_tokens]
    num_word2vec_ere = len(word2vec_tokens)

    # Indices to designate type of ERE
    ere_indexer.get_index('<CAT>Event', add=True)
    ere_indexer.get_index('<CAT>Relation', add=True)
    ere_indexer.get_index('<CAT>Entity', add=True)
    [ere_indexer.get_index(item, add=True) for item in token_set - word2vec_tokens]

    ere_emb_mat = np.zeros((ere_indexer.size, emb_dim), dtype=np.float64)

    # Load corresponding pretrained Word2Vec embeddings
    for iter in range(num_word2vec_ere):
        word = ere_indexer.get_word(iter)
        ere_emb_mat[iter] = word2vec_model.get_vector(word)

    # For tokens which do not appear in Word2Vec vocab, assign a random vector
    for iter in range(num_word2vec_ere, ere_indexer.size):
        ere_emb_mat[iter] = np.random.uniform(min_vec, max_vec, emb_dim)

    word2vec_tokens = set.intersection(stmt_labels, set(word2vec_model.vocab.keys()))

    [stmt_indexer.get_index(item, add=True) for item in word2vec_tokens]
    num_word2vec_stmt = len(word2vec_tokens)

    # Indices to distinguish between non-typing and typing stmts
    stmt_indexer.get_index('<Typing>No', add=True)
    stmt_indexer.get_index('<Typing>Yes', add=True)
    [stmt_indexer.get_index(item, add=True) for item in stmt_labels - word2vec_tokens]

    stmt_emb_mat = np.zeros((stmt_indexer.size, emb_dim), dtype=np.float64)

    for iter in range(num_word2vec_stmt):
        word = stmt_indexer.get_word(iter)
        stmt_emb_mat[iter] = word2vec_model.get_vector(word)

    for iter in range(num_word2vec_stmt, stmt_indexer.size):
        stmt_emb_mat[iter] = np.random.uniform(min_vec, max_vec, emb_dim)

    dill.dump((ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt), open(os.path.join(indexer_dir, "indexers.p"), "wb"))

    return (ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt)

# Convert ERE/stmt labels to indices
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

    # Create adjacency matrices
    # adj_ere_to_stmt_head: (num_eres x num_stmts) adjacency matrix;
    # contains a 1 in position (x, y) if statement y is attached to ERE x at the head/subject (does NOT include typing statements)
    # adj_ere_to_stmt_tail: same as above, but for statements attached to EREs at the tail
    # adj_ere_to_stmt_head: same as above, but for typing statements attached to EREs
    adj_ere_to_stmt_head = np.zeros((ere_mat_ind.size, stmt_mat_ind.size), dtype=bool)
    adj_ere_to_stmt_tail = np.zeros((ere_mat_ind.size, stmt_mat_ind.size), dtype=bool)
    adj_ere_to_type_stmt = np.zeros((ere_mat_ind.size, stmt_mat_ind.size), dtype=bool)

    # Old: for dividing adjacency matrix by node degree
    #inv_sqrt_degree_mat_eres = np.zeros((ere_mat_ind.size, ere_mat_ind.size))
    #inv_sqrt_degree_mat_stmts = np.zeros((stmt_mat_ind.size, stmt_mat_ind.size))

    ere_labels = [[] for _ in range(ere_mat_ind.size)]
    stmt_labels = [[] for _ in range(stmt_mat_ind.size)]

    # Fill in adjacency matrices
    for ere_iter in range(ere_mat_ind.size):
        ere_id = ere_mat_ind.get_word(ere_iter)
        head_stmt_keys = [stmt_mat_ind.get_index(stmt_key, add=False) for stmt_key in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[stmt_key].head_id == ere_id and graph_mix.stmts[stmt_key].tail_id]
        type_stmt_keys = [stmt_mat_ind.get_index(stmt_key, add=False) for stmt_key in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[stmt_key].head_id == ere_id and not graph_mix.stmts[stmt_key].tail_id]
        tail_stmt_keys = [stmt_mat_ind.get_index(stmt_key, add=False) for stmt_key in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[stmt_key].tail_id == ere_id]

        adj_ere_to_stmt_head[ere_iter][head_stmt_keys] = 1
        adj_ere_to_stmt_tail[ere_iter][tail_stmt_keys] = 1
        adj_ere_to_type_stmt[ere_iter][type_stmt_keys] = 1
        #inv_sqrt_degree_mat_eres[ere_iter][ere_iter] = 1 / np.sqrt(len(graph_mix.eres[ere_id].stmt_ids) + 1)

        # Record name/label indices (starting with ERE category)
        ere_labels[ere_iter].append([ere_indexer.get_index('<CAT>' + graph_mix.eres[ere_id].category, add=False)])

        for label in [item for item in graph_mix.eres[ere_id].label if item != 'Relation']:
            temp = []
            if ('_').join(label.split()) in ere_indexer.word_to_index.keys():
                temp.append(ere_indexer.get_index(('_').join(label.split()), add=False))
            else:
                for word in label.split():
                    if word in ere_indexer.word_to_index.keys():
                        temp.append(ere_indexer.get_index(word, add=False))

            if len(temp) > 0:
                ere_labels[ere_iter].append(temp)

    # Record name/label indices (starting with typing/non-typing category
    for stmt_iter in range(stmt_mat_ind.size):
        stmt_id = stmt_mat_ind.get_word(stmt_iter)

        stmt_labels[stmt_iter].append([stmt_indexer.get_index('<Typing>' + ('No' if graph_mix.stmts[stmt_id].tail_id else 'Yes'), add=False)])
        #inv_sqrt_degree_mat_stmts[stmt_iter][stmt_iter] = 1 / np.sqrt(3) if graph_mix.stmts[stmt_id].tail_id else 1 / np.sqrt(2)

        for label in graph_mix.stmts[stmt_id].label:
            temp = []

            # Split by capital letters
            for word in re.findall('[a-zA-Z][^A-Z]*', label):
                if word.lower() in stmt_indexer.word_to_index.keys():
                    temp.append(stmt_indexer.get_index(word.lower(), add=False))

            stmt_labels[stmt_iter].append(temp)

    return {'graph_mix': graph_mix, 'ere_mat_ind': ere_mat_ind, 'stmt_mat_ind': stmt_mat_ind, 'adj_head': adj_ere_to_stmt_head, 'adj_tail': adj_ere_to_stmt_tail, 'adj_type': adj_ere_to_type_stmt, 'ere_labels': ere_labels, 'stmt_labels': stmt_labels, 'num_word2vec_ere': num_word2vec_ere, 'num_word2vec_stmt': num_word2vec_stmt}

# Index graph salads and write them to disk.
def index(data_dir, pre_word2vec_bin, emb_dim, return_freq_cut_set):
    indexed_data_dir = data_dir + '_Indexed'

    if not os.path.exists(indexed_data_dir):
        os.makedirs(indexed_data_dir)

    train_path = os.path.join(data_dir, 'Train')
    val_path = os.path.join(data_dir, 'Val')
    test_path = os.path.join(data_dir, 'Test')

    verify_dir(os.path.join(indexed_data_dir, 'Train'))
    verify_dir(os.path.join(indexed_data_dir, 'Val'))
    verify_dir(os.path.join(indexed_data_dir, 'Test'))

    # Load pretrained Word2Vec model
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(pre_word2vec_bin, binary=True)

    # Create the indexer
    indexer_info = create_indexers_for_corpus(train_path, indexed_data_dir, word2vec_model, emb_dim, return_freq_cut_set)

    # Index graph salads
    for data_iter, path in enumerate([train_path, val_path, test_path]):
        for file_iter, file_name in enumerate(os.listdir(path)):
            _, query, graph_mix, target_graph_id = dill.load(open(os.path.join(path, file_name), 'rb'))

            graph_dict = convert_labels_to_indices(graph_mix, indexer_info)

            query_stmt_indices = [graph_dict['stmt_mat_ind'].get_index(stmt_id, add=False) for stmt_id in query]

            # Find the indices of all EREs captured by statements in the query set.
            query_ere_indices = set.union(*[set([graph_dict['ere_mat_ind'].get_index(graph_dict['graph_mix'].stmts[stmt].head_id, add=False), graph_dict['ere_mat_ind'].get_index(graph_dict['graph_mix'].stmts[stmt].tail_id, add=False)])
                                            if graph_dict['graph_mix'].stmts[stmt].tail_id else set([graph_dict['ere_mat_ind'].get_index(graph_dict['graph_mix'].stmts[stmt].head_id, add=False)]) for stmt in query])

            graph_dict['query_stmts'] = query_stmt_indices
            graph_dict['query_eres'] = query_ere_indices
            graph_dict['target_graph_id'] = target_graph_id

            if data_iter == 0:
                dill.dump(graph_dict, open(os.path.join(indexed_data_dir, 'Train', file_name), 'wb'))
            elif data_iter == 1:
                dill.dump(graph_dict, open(os.path.join(indexed_data_dir, 'Val', file_name), 'wb'))
            else:
                dill.dump(graph_dict, open(os.path.join(indexed_data_dir, 'Test', file_name), 'wb'))

            print(file_iter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/atomko/out_salads", help='Directory containing folders of different mixture types')
    parser.add_argument("--pre_word2vec_bin", type=str, default='/home/atomko/GoogleNews-vectors-negative300.bin', help='Location (abs path) of binary containing pretrained Word2Vec embeds')
    parser.add_argument("--emb_dim", type=int, default=300, help='Index the x most frequent ERE label tokens')
    parser.add_argument("--return_freq_cut_set", type=int, default=50000, help='Index the x most frequent ERE label tokens')

    args = parser.parse_args()
    locals().update(vars(args))

    index(data_dir, pre_word2vec_bin, emb_dim, return_freq_cut_set)

