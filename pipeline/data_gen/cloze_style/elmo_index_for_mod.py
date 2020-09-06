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

# word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/atomko/backup_drive/Google_News_Embeds/GoogleNews-vectors-negative300.bin', binary=True)
# word2vec_vocab = word2vec_model.vocab.keys()
#
# min_vec = np.full(300, np.inf, dtype=np.float32)
# max_vec = -1 * np.full(300, np.inf, dtype=np.float32)
#
# for item in word2vec_vocab:
#     min_vec = np.amin(np.concatenate([min_vec.reshape((1, -1)), word2vec_model.get_vector(item).reshape((1, -1))], axis=0), axis=0)
#     max_vec = np.amax(np.concatenate([max_vec.reshape((1, -1)), word2vec_model.get_vector(item).reshape((1, -1))], axis=0), axis=0)

# tokens = get_tokens('/home/atomko/backup_drive/Data_Gen_Subtask/Mixtures/Train')
# print(len(tokens))

def compute_connectedness(graph):
    """Compute the connectedness of all ERE nodes in a graph"""
    graph.connectedness_one_step = {}
    graph.connectedness_two_step = {}

    for ere_id, ere in graph.eres.items():
        graph.connectedness_one_step[ere_id] = len(ere.neighbor_ere_ids)

        two_step_neighbor_ere_ids = set()
        for neighbor_id in ere.neighbor_ere_ids:
            two_step_neighbor_ere_ids.update(graph.eres[neighbor_id].neighbor_ere_ids)

        two_step_neighbor_ere_ids.update(ere.neighbor_ere_ids)
        two_step_neighbor_ere_ids.discard(ere_id)

        graph.connectedness_two_step[ere_id] = len(two_step_neighbor_ere_ids)

def get_embs(file_path, ont_tok_indexer, num_query_hops):
    if 'self' in file_path:
        graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt = dill.load(open(file_path, 'rb'))

        #assert len([stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if stmt_id in res_stmts]) == 1
        graph_ids = [graph.graph_id]

        #stmt_list = [item for item in graph.eres[ent_ere_id].stmt_ids if graph.stmts[item].tail_id and item in res_stmts]
        #assert len(stmt_list) == 1
        #assert stmt_list[0] == target_event_stmt
        assert target_event_stmt in res_stmts
        query_side_stmts = {stmt_id for stmt_id in graph.stmts.keys() if stmt_id not in res_stmts}
        cand_side_stmts = res_stmts
    else:
        graph, query_stmts, ent_ere_id, ent_ere_ids, target_event_stmt = dill.load(open(file_path, 'rb'))
        graph_ids = list({stmt_id[:26] for stmt_id in graph.eres[ent_ere_id].stmt_ids})

        target_graph_id = graph.stmts[list(query_stmts)[0]].graph_id

        #assert len([stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if graph.stmts[stmt_id].graph_id != target_graph_id]) == 1
        assert len(graph_ids) == 2

        query_side_stmts = {stmt_id for stmt_id in graph.stmts.keys() if graph.stmts[stmt_id].graph_id != graph.stmts[target_event_stmt].graph_id}
        cand_side_stmts = {stmt_id for stmt_id in graph.stmts.keys() if graph.stmts[stmt_id].graph_id == graph.stmts[target_event_stmt].graph_id}
        assert set.union(query_side_stmts, cand_side_stmts) == set(graph.stmts.keys())

    seen_eres = set()
    curr_eres = {ent_ere_id}

    curr_num_hops = 0

    stmt_id_sets = []

    while curr_num_hops < num_query_hops and len(curr_eres) > 0:
        seen_eres.update(curr_eres)

        stmt_ids = set.union(*[set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (stmt_id in query_side_stmts)]) for ere_id in curr_eres])

        if curr_num_hops > 0:
            stmt_ids -= set.union(*stmt_id_sets[:curr_num_hops])

        for stmt_id in stmt_ids:
            curr_eres.update([graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id])

        if stmt_ids:
            stmt_id_sets.append(stmt_ids)

        curr_eres -= seen_eres
        curr_num_hops += 1

    assert len(set.intersection(set.union(*stmt_id_sets), query_stmts)) == len(set.union(*stmt_id_sets))

    query_stmts = set.union(*stmt_id_sets)
    query_eres = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in query_stmts])
    query_stmts.update(set.union(*[set([stmt_id for stmt_id in graph.eres[query_ere].stmt_ids if not graph.stmts[stmt_id].tail_id]) for query_ere in query_eres]))

    graph_map = defaultdict(lambda: defaultdict(dict))

    for graph_id in graph_ids:
        h5_main_path = os.path.join('/home/atomko/H5_Samples', graph_id + '.h5')

        h5_obj = h5py.File(h5_main_path, 'r')

        graph_map[graph_id]['graph'] = dill.load(open(os.path.join('/home/atomko/backup_drive/Summer_2020/Graph_Singles', graph_id + '.p'), 'rb'))
        graph_map[graph_id]['non_type_emb'] = h5_obj['non_type_group']['Non-typing Statement Embeddings']
        graph_map[graph_id]['type_group'] = h5_obj['type_group']
        graph_map[graph_id]['non_type_stmt_list'] = dill.load(open('/home/atomko/H5_Samples/non_type_stmt_names_' + graph_id + '.p', 'rb'))
        graph_map[graph_id]['type_to_non_map'] = dill.load(open('/home/atomko/H5_Samples/type_to_non_map_' + graph_id + '.p', 'rb'))

    non_type_emb = np.concatenate([graph_map[graph_id]['non_type_emb'] for graph_id in graph_ids], axis=0)
    non_type_stmt_list = [item for sublist in [graph_map[graph_id]['non_type_stmt_list'] for graph_id in graph_ids] for item in sublist]
    type_to_non_map = dict()

    for graph_id in graph_ids:
        type_to_non_map.update(graph_map[graph_id]['type_to_non_map'])

    non_type_stmt_map = dict()
    non_type_stmt_ont = dict()

    non_type_stmt_to_ind = dict()

    for iter, item in enumerate(non_type_stmt_list):
        non_type_stmt_to_ind[item] = iter

    for item in {sub_item for sub_item in graph.stmts.keys() if graph.stmts[sub_item].tail_id}:
        head_ere_id = graph.stmts[item].head_id

        head_types = [stmt_id for stmt_id in graph.eres[head_ere_id].stmt_ids if not graph.stmts[stmt_id].tail_id]
        assert len(head_types) == 1

        head_type_map = [type_to_non_map[head_types[0].split('_dup')[0]][key] for key in sorted(list(type_to_non_map[head_types[0].split('_dup')[0]].keys()))]

        head_iter = None

        for sub_iter in range(len(head_type_map)):
            if item in head_type_map[sub_iter]:
                head_iter = sub_iter
                break

        comp_graph = graph_map[graph.stmts[item].graph_id]['graph']
        tail_types = [stmt_id for stmt_id in comp_graph.eres[comp_graph.stmts[item].tail_id].stmt_ids if not comp_graph.stmts[stmt_id].tail_id]

        assert len(tail_types) == 1

        tail_type_map = [type_to_non_map[tail_types[0].split('_dup')[0]][key] for key in sorted(list(type_to_non_map[tail_types[0].split('_dup')[0]].keys()))]

        tail_iter = None

        for sub_iter in range(len(tail_type_map)):
            if item in tail_type_map[sub_iter]:
                tail_iter = sub_iter
                break

        head_labels = []

        for label in re.sub("[._]", " ", graph.stmts[head_types[0]].raw_label).split():
            head_labels += [word.lower() for word in re.findall('[a-zA-Z][^A-Z]*', label)]

        label = re.sub("[._]", " ", graph.stmts[item].raw_label).split()[-1]
        role_labels = [word.lower() for word in re.findall('[a-zA-Z][^A-Z]*', label)]

        tail_labels = []

        for label in re.sub("[._]", " ", comp_graph.stmts[tail_types[0]].raw_label).split():
            tail_labels += [word.lower() for word in re.findall('[a-zA-Z][^A-Z]*', label)]

        non_type_stmt_ont[item] = (head_labels, role_labels, tail_labels)

        if graph.eres[head_ere_id].category == 'Event':
            non_type_stmt_map[item] = (graph_map[graph.eres[head_ere_id].graph_id]['type_group'][re.sub('/', '_', head_types[0].split('_dup')[0])][head_iter], non_type_emb[non_type_stmt_to_ind[item]])
        elif graph.eres[head_ere_id].category == 'Relation':
            non_type_stmt_map[item] = (graph_map[graph.eres[head_ere_id].graph_id]['type_group'][re.sub('/', '_', head_types[0].split('_dup')[0])][head_iter], graph_map[comp_graph.graph_id]['type_group'][re.sub('/', '_', tail_types[0].split('_dup')[0])][tail_iter])

    assert (set(non_type_stmt_map.keys()) == {stmt_id for stmt_id in graph.stmts.keys() if graph.stmts[stmt_id].tail_id})

    arg_stmt_mat_ind = Indexer()
    arg_stmt_mat_ind.word_to_index = {}
    arg_stmt_mat_ind.index_to_word = {}
    arg_stmt_mat_ind.size = 0

    for stmt_id in {item for item in graph.stmts.keys() if graph.stmts[item].tail_id}:
        arg_stmt_mat_ind.get_index(stmt_id, add=True)

    evr_meet_adj = np.zeros((arg_stmt_mat_ind.size, arg_stmt_mat_ind.size), dtype=np.uint8)
    ent_meet_adj = np.zeros((arg_stmt_mat_ind.size, arg_stmt_mat_ind.size), dtype=np.uint8)

    for stmt_iter in range(arg_stmt_mat_ind.size):
        stmt_id = arg_stmt_mat_ind.get_word(stmt_iter)
        evr_meet_keys = [arg_stmt_mat_ind.get_index(stmt_key, add=False) for stmt_key in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[stmt_key].tail_id and stmt_id != stmt_key]
        ent_meet_keys = [arg_stmt_mat_ind.get_index(stmt_key, add=False) for stmt_key in graph.eres[graph.stmts[stmt_id].tail_id].stmt_ids if graph.stmts[stmt_key].tail_id and stmt_id != stmt_key]

        evr_meet_adj[stmt_iter][evr_meet_keys] = 1
        ent_meet_adj[stmt_iter][ent_meet_keys] = 1

    non_type_final_emb = np.zeros((arg_stmt_mat_ind.size, 3, (1024 * 2)), dtype=np.float32)

    non_type_ont_labels = []

    for stmt_iter in range(arg_stmt_mat_ind.size):
        stmt_id = arg_stmt_mat_ind.get_word(stmt_iter)

        ont_labels = non_type_stmt_ont[stmt_id]

        temp_set = []

        for label_iter, label_set in enumerate(ont_labels):
            sub_temp = []

            if label_iter in [0, 2]:
                sub_temp.append(ont_tok_indexer.get_index('<Typing>Yes', add=False))
            elif label_iter == 1:
                sub_temp.append(ont_tok_indexer.get_index('<Typing>No', add=False))

            temp_set.append(sub_temp + [ont_tok_indexer.get_index(tok, add=False) for tok in label_set if tok in ont_tok_indexer.word_to_index.keys()])

        non_type_ont_labels.append(temp_set)
        non_type_final_emb[stmt_iter] = np.concatenate([item for item in non_type_stmt_map[stmt_id]], -1)

    seen_eres = set()
    curr_eres = {ent_ere_id}

    curr_num_hops = 0

    stmt_id_sets = []

    while len(curr_eres) > 0:
        seen_eres.update(curr_eres)

        stmt_ids = set.union(*[set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (stmt_id in cand_side_stmts)]) for ere_id in curr_eres])

        if curr_num_hops > 0:
            stmt_ids -= set.union(*stmt_id_sets[:curr_num_hops])

        for stmt_id in stmt_ids:
            curr_eres.update([graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id])

        if stmt_ids:
            stmt_id_sets.append(stmt_ids)

        curr_eres -= seen_eres
        curr_num_hops += 1

    if len(stmt_id_sets) == 6:
        stmt_weight_map = {0 : .35, 1 : .25, 2 : .20, 3 : .10, 4 : .06, 5 : .04}
    elif len(stmt_id_sets) == 5:
        stmt_weight_map = {0 : .358, 1 : .258, 2 : .208, 3 : .108, 4 : .068}
    elif len(stmt_id_sets) == 4:
        stmt_weight_map = {0 : .375, 1 : .275, 2 : .225, 3 : .125}
    elif len(stmt_id_sets) == 3:
        stmt_weight_map = {0 : .4166666666666, 1 : .3166666666666, 2 : .2666666666666}
    elif len(stmt_id_sets) == 2:
        stmt_weight_map = {0 : .55, 1 : .45}
    elif len(stmt_id_sets) == 1:
        stmt_weight_map = {0 : 1.0}

    arg_cand_stmt_weights = np.zeros(arg_stmt_mat_ind.size, dtype=np.float32)

    for stmt_list_iter, stmt_list in enumerate(stmt_id_sets):
        list_total = len(stmt_list)

        for stmt_id in stmt_list:
            stmt_ind = arg_stmt_mat_ind.get_index(stmt_id, add=False)

            arg_cand_stmt_weights[stmt_ind] = stmt_weight_map[stmt_list_iter] / float(list_total)

    assert np.where(arg_cand_stmt_weights == 0)[0].size == len({item for item in query_side_stmts if graph.stmts[item].tail_id})

    seen_eres = set()
    curr_eres = {ent_ere_id}

    curr_num_hops = 0

    stmt_id_sets = []

    while len(curr_eres) > 0:
        seen_eres.update(curr_eres)

        stmt_ids = set.union(*[set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (stmt_id in query_side_stmts)]) for ere_id in curr_eres])

        if curr_num_hops > 0:
            stmt_ids -= set.union(*stmt_id_sets[:curr_num_hops])

        for stmt_id in stmt_ids:
            curr_eres.update([graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id])

        if stmt_ids:
            stmt_id_sets.append(stmt_ids)

        curr_eres -= seen_eres
        curr_num_hops += 1

    if len(stmt_id_sets) == 6:
        stmt_weight_map = {0: .35, 1: .25, 2: .20, 3: .10, 4: .06, 5: .04}
    elif len(stmt_id_sets) == 5:
        stmt_weight_map = {0: .358, 1: .258, 2: .208, 3: .108, 4: .068}
    elif len(stmt_id_sets) == 4:
        stmt_weight_map = {0: .375, 1: .275, 2: .225, 3: .125}
    elif len(stmt_id_sets) == 3:
        stmt_weight_map = {0: .4166666666666, 1: .3166666666666, 2: .2666666666666}
    elif len(stmt_id_sets) == 2:
        stmt_weight_map = {0: .55, 1: .45}
    elif len(stmt_id_sets) == 1:
        stmt_weight_map = {0: 1.0}

    arg_query_stmt_weights = np.zeros(arg_stmt_mat_ind.size, dtype=np.float32)

    for stmt_list_iter, stmt_list in enumerate(stmt_id_sets):
        list_total = len(stmt_list)

        for stmt_id in stmt_list:
            stmt_ind = arg_stmt_mat_ind.get_index(stmt_id, add=False)

            arg_query_stmt_weights[stmt_ind] = stmt_weight_map[stmt_list_iter] / float(list_total)

    assert np.where(arg_query_stmt_weights == 0)[0].size == len({item for item in cand_side_stmts if graph.stmts[item].tail_id})

    compute_connectedness(graph)
    ent_conn = graph.connectedness_two_step[ent_ere_id]
    named_ent = [item for item in ''.join(graph.eres[ent_ere_id].label) if item.isupper()]

    return arg_stmt_mat_ind, evr_meet_adj, ent_meet_adj, non_type_final_emb, non_type_ont_labels, [arg_stmt_mat_ind.get_index(stmt_id, add=False) for stmt_id in query_stmts if graph.stmts[stmt_id].tail_id], arg_stmt_mat_ind.get_index(target_event_stmt, add=False), \
           [arg_stmt_mat_ind.get_index(stmt_id, add=False) for stmt_id in query_side_stmts if graph.stmts[stmt_id].tail_id], [arg_stmt_mat_ind.get_index(stmt_id, add=False) for stmt_id in cand_side_stmts if graph.stmts[stmt_id].tail_id], arg_query_stmt_weights, arg_cand_stmt_weights, ent_conn, named_ent
    #print(file_iter)