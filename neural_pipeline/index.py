import dill
import os
import re
import shutil
import random
import torch
import argparse
import time
from collections import Counter, defaultdict
from adapted_data_to_input import read_graph, Indexer
import os
import json


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

def index_and_partition(seed_to_subgraph_map, indexed_data_dir, indexer_dir):
    if not os.path.exists(indexed_data_dir):
        os.makedirs(indexed_data_dir)

    indexer_info = dill.load(open(os.path.join(indexer_dir, 'indexers.p'), 'rb'))

    for group_name in seed_to_subgraph_map.keys():
        if not os.path.exists(os.path.join(indexed_data_dir, group_name)):
            os.makedirs(os.path.join(indexed_data_dir, group_name))

        for seed_path in seed_to_subgraph_map[group_name].keys():
            if not os.path.exists(os.path.join(indexed_data_dir, group_name, seed_path.split('_')[-2][-4:])):
                os.makedirs(os.path.join(indexed_data_dir, group_name, seed_path.split('_')[-2][-4:]))

            seed_json = json.load(open(seed_path, 'r'))

            for iter, subgraph_path in enumerate(sorted(seed_to_subgraph_map[group_name][seed_path], key=lambda x: int(x.split('_')[-1].split('.json')[0]))):
                graph = seed_to_subgraph_map[group_name][seed_path][subgraph_path]
                graph_dict = convert_labels_to_indices(graph, indexer_info)

                query = [(graph.graph_id + '_' + item) for item in seed_json['support'][iter]['statements']]
                query_stmt_indices = [graph_dict['stmt_mat_ind'].get_index(stmt, add=False) for stmt in query]
                query_ere_indices = set.union(*[{graph_dict['ere_mat_ind'].get_index(graph.stmts[stmt_id].head_id, add=False), graph_dict['ere_mat_ind'].get_index(graph.stmts[stmt_id].tail_id, add=False)}
                                              if graph.stmts[stmt_id].tail_id else {graph_dict['ere_mat_ind'].get_index(graph.stmts[stmt_id].head_id, add=False)} for stmt_id in query])

                graph_dict['query_stmts'] = query_stmt_indices
                graph_dict['query_eres'] = query_ere_indices

                dill.dump(graph_dict, open(os.path.join(indexed_data_dir, group_name, seed_path.split('_')[-2][-4:], graph_dict['graph_mix'].graph_id + '.p'), 'wb'))

def map_subgraph_to_seed(data_dir, graphs_folder, hypothesis_folder):
    group_dicts = dict()

    for group_name in os.listdir(data_dir):
        group_data_dir = os.path.join(data_dir, group_name)
        graph_dir = os.path.join(group_data_dir, graphs_folder)
        seed_dir = os.path.join(group_data_dir, hypothesis_folder)

        subgraph_map = dict()

        for graph_name in os.listdir(graph_dir):
            subgraph_map[graph_name] = [os.path.join(graph_dir, graph_name, item) for item in os.listdir(os.path.join(graph_dir, graph_name))]

        seed_subgraph_map = defaultdict(lambda: defaultdict())

        for seed_key in sorted(list(subgraph_map.keys())):
            seed_name = None

            for item in sorted(os.listdir(seed_dir)):
                if seed_key in item:
                    seed_name = item

                    break

            seed_path = os.path.join(seed_dir, seed_name)

            for subgraph_path in sorted(subgraph_map[seed_key], key=lambda x: int(x.split('_')[-1].split('.json')[0])):
                json_obj = json.load(open(subgraph_path))["theGraph"]
                graph = read_graph(re.sub('/', '_', subgraph_path.split(graphs_folder + ('/' if not graphs_folder.endswith('/') else ''))[1].split('.json')[0]), json_obj)
                seed_subgraph_map[seed_path][subgraph_path] = graph

        group_dicts[group_name] = seed_subgraph_map

    return group_dicts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/aida-utexas/data", help='Directory containing group dirs (e.g., LDC_2.LDC_2, OPERA_4_Colorado)')
    parser.add_argument("--graphs_folder", type=str, default="subgraph", help='Name of subdirectory containing subgraphs')
    parser.add_argument("--hypothesis_folder", type=str, default="cluster_seeds", help='Name of subdirectory containing seed files')
    parser.add_argument("--indexed_data_dir", type=str, default='/aida-utexas/data_indexed', help='Directory of indexed data')
    parser.add_argument("--indexer_dir", type=str, default='/aida-utexas/neural_pipeline', help='Directory containing indexers file')

    args = parser.parse_args()
    locals().update(vars(args))

    seed_to_subgraph_map = map_subgraph_to_seed(data_dir, graphs_folder, hypothesis_folder)

    print('Indexing files----------')

    index_and_partition(seed_to_subgraph_map, indexed_data_dir, indexer_dir)

    print('Indexing finished: files in directory ' + indexed_data_dir)