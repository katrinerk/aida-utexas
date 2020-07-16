""" Various facilities for data handling and training.

Author: Su Wang; 2019.
"""

import os
import random
from copy import deepcopy

import dill
import numpy as np
import torch


def pad(indices):
    """Pad a list of lists to the longest sublist. Return as a list of lists (easy manip. in run_batch)."""
    padded_indices = []
    max_len = max(len(sub_indices) for sub_indices in indices)
    padded_indices = [sub_indices[:max_len]
                      if len(sub_indices) >= max_len
                      else sub_indices + [0] * (max_len - len(sub_indices))
                      for sub_indices in indices]
    return padded_indices


def get_candidates(batch):
    query_stmts_list = [graph_dict['query_stmts'] for graph_dict in batch]
    query_eres_list = [graph_dict['query_eres'] for graph_dict in batch]
    stmt_mat_ind_list = [graph_dict['stmt_mat_ind'] for graph_dict in batch]
    ere_mat_ind_list = [graph_dict['ere_mat_ind'] for graph_dict in batch]
    graph_mix_list = [graph_dict['graph_mix'] for graph_dict in batch]
    candidate_ids_list = [[] for _ in batch]

    for iter, query_eres in enumerate(query_eres_list):
        for ere_iter in query_eres:
            neigh_stmts = graph_mix_list[iter].eres[ere_mat_ind_list[iter].get_word(ere_iter)].stmt_ids
            candidate_ids_list[iter] += [stmt_mat_ind_list[iter].get_index(neigh_stmt, add=False) for neigh_stmt in neigh_stmts if stmt_mat_ind_list[iter].get_index(neigh_stmt, add=False) not in query_stmts_list[iter]]

        candidate_ids_list[iter] = list(set(candidate_ids_list[iter]))

    return candidate_ids_list


class DataIterator:

    def __init__(self, data_dirs, batch_size, evaluate):
        """`data_dir`: under which we have .p saved graph files."""
        self.data_dirs = data_dirs
        self.file_paths = [item for sublist in [[os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)] for data_dir in data_dirs] for item in sublist]
        self.shuffle()
        self.size = len(self.file_paths)
        self.cursor = 0
        self.epoch = 0
        self.batch_size = batch_size
        self.evaluate = evaluate

    def shuffle(self):
        """Shuffle reading order."""
        self.file_paths = sorted(self.file_paths)
        random.seed(5)
        random.shuffle(self.file_paths)
        #self.file_paths = sorted(self.file_paths, key=lambda x: int(x.split('.p')[0].split('json')[-1]))
        #self.file_paths = sorted(self.file_paths, key=lambda x: (int(x.split('/')[-1][1:4]), int(x.split('.p')[0].split('_')[-1])))

    def next_batch(self):
        """Read one saved data graph."""
        if self.cursor >= self.size:
            self.cursor = 0
            self.epoch += 1
        graph_dicts = [dill.load(open(self.file_paths[item], "rb")) for item in range(self.cursor, self.cursor + self.batch_size)]
        candidate_ids_list, label_inputs_list = format_inputs(graph_dicts, self.evaluate)

        self.cursor += self.batch_size

        for iter in range(len(graph_dicts)):
            graph_dicts[iter]['candidates'] = candidate_ids_list[iter]
            graph_dicts[iter]['label_inputs'] = label_inputs_list[iter]

        return graph_dicts


def get_random_gold_label(batch, prediction_list, use_high_ranked_gold):
    """Return a randomly selected correct label (its index) for teacher enforce."""
    correct_indices = [[index for index, label in enumerate(batch[iter]['label_inputs']) if label == 1] for iter in range(len(batch))]

    if use_high_ranked_gold:
        correct_indices = [[correct_indices[iter][int(torch.argmax(prediction_list[iter][correct_indices[iter]]))]] for iter in range(len(batch))]

    random_correct_indices = [random.choice(correct_indices[iter]) if len(correct_indices[iter]) != 0 else 0 for iter in range(len(batch))]
    return random_correct_indices


def get_tensor_labels(batch, device=torch.device("cpu")):
    """Format [...] labels to torch tensors (Float)."""
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
    """Update state by inserting selected candidate (ere and attr.) to query set."""
    for iter, selected_index in enumerate(selected_indices):
        batch[iter]['query_stmts'] = np.insert(batch[iter]['query_stmts'], len(batch[iter]['query_stmts']), batch[iter]['candidates'][selected_index])

    temp = [[batch[iter]['graph_mix'].stmts[batch[iter]['stmt_mat_ind'].get_word(batch[iter]['candidates'][selected_index])].head_id, batch[iter]['graph_mix'].stmts[batch[iter]['stmt_mat_ind'].get_word(batch[iter]['candidates'][selected_index])].tail_id] for iter, selected_index in enumerate(selected_indices)]

    for iter, item in enumerate(temp):
        if not item[1]:
            temp[iter] = item[0:1]

    sel_vals = [batch[iter]['candidates'][selected_index] for iter, selected_index in enumerate(selected_indices)]

    for iter, selected_index in enumerate(selected_indices):
        del batch[iter]['candidates'][selected_index]

    if not evaluate:
        for iter, selected_index in enumerate(selected_indices):
            if len(batch[iter]['label_inputs']) != 0:
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


def get_stmt_ids(gcn_info, predicted_ids):
    """Return statement ids given predicted ere ids."""
    stmt_ids = []
    for predicted_id in predicted_ids:
        stmt_ids += gcn_info[predicted_id]["Stmts"]
    return list(set(stmt_ids))
