""" Various facilities for data handling and training.

Author: Su Wang; 2019; modified by Alexander Tomkovich
"""

from collections import defaultdict
from copy import deepcopy
import dill
import numpy as np
import random
import os
import torch
import collections
import re


def get_candidates(batch, evaluate):
    query_stmts_list = [graph_dict['query_stmts'] for graph_dict in batch]
    query_eres_list = [graph_dict['query_eres'] for graph_dict in batch]
    stmt_mat_ind_list = [graph_dict['stmt_mat_ind'] for graph_dict in batch]
    ere_mat_ind_list = [graph_dict['ere_mat_ind'] for graph_dict in batch]
    graph_mix_list = [graph_dict['graph_mix'] for graph_dict in batch]
    candidate_ids_list = [[] for _ in batch]
    label_inputs_list = [[] for _ in batch]

    if not evaluate:
        target_graph_id_list = [graph_dict['target_graph_id'] for graph_dict in batch]

    for iter, query_eres in enumerate(query_eres_list):
        for ere_iter in query_eres:
            neigh_stmts = graph_mix_list[iter].eres[ere_mat_ind_list[iter].get_word(ere_iter)].stmt_ids
            candidate_ids_list[iter] += [stmt_mat_ind_list[iter].get_index(neigh_stmt, add=False) for neigh_stmt in neigh_stmts if stmt_mat_ind_list[iter].get_index(neigh_stmt, add=False) not in query_stmts_list[iter]]

        candidate_ids_list[iter] = list(set(candidate_ids_list[iter]))

        if not evaluate:
            [label_inputs_list[iter].append(1 if graph_mix_list[iter].stmts[stmt_mat_ind_list[iter].get_word(candidate_id)].graph_id == target_graph_id_list[iter] else 0) for candidate_id in candidate_ids_list[iter]]

    return candidate_ids_list, label_inputs_list


class DataIterator:

    def __init__(self, data_dir, batch_size):
        self.file_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]
        self.cursor = 0
        self.epoch = 0
        self.shuffle()
        self.size = len(self.file_paths)
        self.batch_size = batch_size

    def shuffle(self):
        self.file_paths = sorted(self.file_paths)
        random.seed(self.epoch)
        random.shuffle(self.file_paths)

    def next_batch(self):
        if self.cursor >= self.size:
            self.cursor = 0
            self.epoch += 1

        graph_dicts = [dill.load(open(self.file_paths[item], "rb")) for item in range(self.cursor, self.cursor + self.batch_size)]
        candidate_ids_list, label_inputs_list = get_candidates(graph_dicts, False)

        self.cursor += self.batch_size

        for iter in range(len(graph_dicts)):
            graph_dicts[iter]['candidates'] = candidate_ids_list[iter]
            graph_dicts[iter]['label_inputs'] = label_inputs_list[iter]

        return graph_dicts


def get_random_gold_label(batch, prediction_list, use_high_ranked_gold):
    correct_indices = [[index for index, label in enumerate(batch[iter]['label_inputs']) if label == 1] for iter in range(len(batch))]

    if use_high_ranked_gold:
        correct_indices = [[correct_indices[iter][int(torch.argmax(prediction_list[iter][correct_indices[iter]]))]] for iter in range(len(batch))]

    random_correct_indices = [random.choice(correct_indices[iter]) for iter in range(len(batch))]
    return random_correct_indices


def get_tensor_labels(batch, device=torch.device("cpu")):
    return [torch.FloatTensor(np.array(deepcopy(batch[iter]['label_inputs']))).to(device) for iter in range(len(batch))]


def multi_correct_nll_loss(predictions, trues, device=torch.device("cpu")):
    """(Batch average) negative Loglikelihood with multiple correct answers (Durret/13)."""
    nll = torch.Tensor([]).to(device)

    for prediction, true in zip(predictions, trues):
        masked_log_probability = torch.mul(prediction, true)
        log_probabilities = masked_log_probability[torch.where(masked_log_probability != 0)[0].squeeze()]#masked_log_probability[masked_log_probability.nonzero().squeeze()]
        if len(log_probabilities.shape) == 0:
            log_probabilities = log_probabilities.unsqueeze(0)
        log_sum_probability = (-1 * torch.logsumexp(log_probabilities, dim=0)).unsqueeze(0)
        if log_sum_probability == float("inf"):
            continue
        nll = torch.cat([nll, log_sum_probability])

    if len(nll) == 0:
        nll = torch.Tensor([0]).to(device)
    return nll.mean()


def next_state(batch, selected_indices, evaluate):
    for iter, selected_index in enumerate(selected_indices):
        batch[iter]['query_stmts'] = np.insert(batch[iter]['query_stmts'], len(batch[iter]['query_stmts']), batch[iter]['candidates'][selected_index])

    temp = []

    for iter, selected_index in enumerate(selected_indices):
        cand_stmt = batch[iter]['graph_mix'].stmts[batch[iter]['stmt_mat_ind'].get_word(batch[iter]['candidates'][selected_index])]

        temp.append([cand_stmt.head_id, cand_stmt.tail_id])

    for iter, item in enumerate(temp):
        if not item[1]:
            temp[iter] = item[0:1]

    sel_vals = [batch[iter]['candidates'][selected_index] for iter, selected_index in enumerate(selected_indices)]

    for iter, selected_index in enumerate(selected_indices):
        del batch[iter]['candidates'][selected_index]

    if not evaluate:
        for iter, selected_index in enumerate(selected_indices):
            if len(batch[iter]['label_inputs']) != 0:##TO-DO
                del batch[iter]['label_inputs'][selected_index]

    set_diffs = [(set([batch[iter]['ere_mat_ind'].get_index(item, add=False) for item in temp[iter]]) - set(batch[iter]['query_eres'])) for iter in range(len(batch))]

    for iter, set_diff in enumerate(set_diffs):
        if set_diff:
            for ere_iter in set_diff:
                stmt_neighs = torch.cat((torch.nonzero(batch[iter]['adj_head'][ere_iter]),
                                         torch.nonzero(batch[iter]['adj_tail'][ere_iter]),
                                         torch.nonzero(batch[iter]['adj_type'][ere_iter])), dim=0).reshape((-1))
                stmt_neighs = [stmt_neigh for stmt_neigh in stmt_neighs[stmt_neighs != sel_vals[iter]].tolist() if
                               stmt_neigh not in batch[iter]['candidates']]
                batch[iter]['candidates'] += stmt_neighs
                if not evaluate:
                    batch[iter]['label_inputs'] = batch[iter]['label_inputs'] + [1 if batch[iter]['graph_mix'].stmts[batch[iter]['stmt_mat_ind'].get_word(stmt_iter)].graph_id == batch[iter]['target_graph_id'] else 0 for stmt_iter in stmt_neighs]

        batch[iter]['query_eres'] = set.union(batch[iter]['query_eres'], set_diff)
