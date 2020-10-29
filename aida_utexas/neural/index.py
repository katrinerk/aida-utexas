import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import dill
import torch

from aida_utexas import util
from aida_utexas.neural.gen_single_doc_graphs import read_graph, Indexer
import numpy as np

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

            if label == 'TopicFIller':
                label = 'TopicFiller'

            if label.isupper():
                if label in stmt_indexer.word_to_index.keys():
                    temp.append(stmt_indexer.get_index(label, add=False))
            else:
                all_caps = list(re.finditer('[A-Z]{2,}', label))
                assert len(all_caps) in [0, 1]

                all_caps_word = None

                if len(all_caps) > 0:
                    span = all_caps[0].span()
                    assert span[1] == len(label)
                    label = label[:span[0]]
                    word = all_caps[0].group(0)

                    if word in stmt_indexer.word_to_index.keys():
                        all_caps_word = word

                # Split by capital letters
                for word in re.findall('[a-zA-Z][^A-Z]*', label):
                    if word.isupper() and word in stmt_indexer.word_to_index.keys():
                        temp.append(stmt_indexer.get_index(word, add=False))
                    elif word.lower() in stmt_indexer.word_to_index.keys():
                        temp.append(stmt_indexer.get_index(word.lower(), add=False))

                if all_caps_word is not None:
                    temp.append(stmt_indexer.get_index(all_caps_word, add=False))

            stmt_labels[stmt_iter].append(temp)

    return {'graph_mix': graph_mix, 'ere_mat_ind': ere_mat_ind, 'stmt_mat_ind': stmt_mat_ind, 'adj_head': adj_ere_to_stmt_head, 'adj_tail': adj_ere_to_stmt_tail, 'adj_type': adj_ere_to_type_stmt, 'ere_labels': ere_labels, 'stmt_labels': stmt_labels, 'num_word2vec_ere': num_word2vec_ere, 'num_word2vec_stmt': num_word2vec_stmt}


def index_and_partition(seed_subgraph_map, output_dir, indexer_path):
    with open(str(indexer_path), 'rb') as fin:
        indexer_info = dill.load(fin)

    for seed_path, subgraph_list in seed_subgraph_map.items():
        sin_name = seed_path.stem.split('_')[0]
        output_subdir = output_dir / sin_name
        output_subdir.mkdir(exist_ok=True, parents=True)

        with open(str(seed_path), 'r') as fin:
            seed_json = json.load(fin)

        for idx, graph in enumerate(subgraph_list):
            graph_dict = convert_labels_to_indices(graph, indexer_info)

            query = [(graph.graph_id + '_' + item) for item in seed_json['support'][idx]['statements']]
            query_stmt_indices = [graph_dict['stmt_mat_ind'].get_index(stmt_id, add=False) for stmt_id in query]
            # Find the indices of all EREs captured by statements in the query set.
            query_ere_indices = set.union(*[set([graph_dict['ere_mat_ind'].get_index(graph_dict['graph_mix'].stmts[stmt].head_id, add=False), graph_dict['ere_mat_ind'].get_index(graph_dict['graph_mix'].stmts[stmt].tail_id, add=False)])
                                            if graph_dict['graph_mix'].stmts[stmt].tail_id else set([graph_dict['ere_mat_ind'].get_index(graph_dict['graph_mix'].stmts[stmt].head_id, add=False)]) for stmt in query])

            graph_dict['query_stmts'] = query_stmt_indices
            graph_dict['query_eres'] = query_ere_indices
            graph_dict['target_graph_id'] = None
            graph_dict['noisy_merge_points'] = None

            with open(output_subdir / (graph_dict['graph_mix'].graph_id + '.p'), 'wb') as fout:
                dill.dump(graph_dict, fout)


def map_subgraph_to_seed(subgraph_dir: Path, seed_dir: Path):
    seed_subgraph_map = defaultdict(list)

    for seed_path in sorted(seed_dir.glob('*_seeds.json')):
        sin_name = seed_path.stem.split('_')[0]
        subgraph_dir_for_sin = subgraph_dir / sin_name

        for subgraph_file in sorted(subgraph_dir_for_sin.glob('*.json'), key=lambda p: int(p.stem.split('_')[1])):
            with open(str(subgraph_file), 'r') as fin:
                graph_json = json.load(fin)["theGraph"]
            graph_name = f'{sin_name}_{subgraph_file.stem}'
            seed_subgraph_map[seed_path].append(read_graph(graph_name, graph_json))

    return seed_subgraph_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", help='path to the working directory.')
    parser.add_argument('--subgraph_dir', default='subgraph',
                        help='name of the subdirectory in working_dir containing subgraphs')
    parser.add_argument('--seed_dir', default='cluster_seeds',
                        help='name of the subdirectory in working_dir containing cluster seeds')
    parser.add_argument('--indexed_data_dir', default='data_indexed',
                        help='name of the subdirectory in working_dir to write indexed data')
    parser.add_argument("--indexer_path", default='resources/indexer.p',
                        help='path to the indexer file.')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    working_dir = util.get_input_path(args.working_dir)

    subgraph_dir = util.get_input_path(working_dir / args.subgraph_dir)
    seed_dir = util.get_input_path(working_dir / args.seed_dir)
    output_dir = util.get_output_dir(working_dir / args.indexed_data_dir,
                                     overwrite_warning=not args.force)

    indexer_path = str(util.get_input_path(args.indexer_path))

    locals().update(vars(args))

    seed_subgraph_map = map_subgraph_to_seed(subgraph_dir, seed_dir)

    print('\nIndexing files ...')

    index_and_partition(seed_subgraph_map, output_dir, indexer_path)

    print(f'\nIndexing finished: indexed data in directory {output_dir}')


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    main()
