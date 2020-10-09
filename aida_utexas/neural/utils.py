""" Various facilities for data handling and training.

Author: Su Wang; 2019.
"""

import os
import re
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
        log_probabilities = masked_log_probability[torch.where(masked_log_probability != 0)[0].squeeze()] # masked_log_probability[masked_log_probability.nonzero().squeeze()]
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
                stmt_neighs = torch.cat((torch.nonzero(batch[iter]['adj_head'][ere_iter], as_tuple=False),
                                         torch.nonzero(batch[iter]['adj_tail'][ere_iter], as_tuple=False),
                                         torch.nonzero(batch[iter]['adj_type'][ere_iter], as_tuple=False)), dim=0).reshape((-1))
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


def select_valid_hypothesis(batch, prediction):
    """Select the index of the highest-scoring valid candidate stmt"""
    graph_mix = batch[0]['graph_mix']
    # get the indices that sort the prediction tensor
    sorted_pred_indices = torch.argsort(prediction[0], descending=True)
    query_stmts = {batch[0]['stmt_mat_ind'].get_word(item) for item in batch[0]['query_stmts']}

    # Find all event EREs with 'Attacker' statements in the query set
    query_events_w_atk = {graph_mix.stmts[stmt_id].head_id for stmt_id in query_stmts if '_Attacker' in graph_mix.stmts[stmt_id].raw_label}
    # Find all event EREs with 'Target' statements in the query set
    query_events_w_trg = {graph_mix.stmts[stmt_id].head_id for stmt_id in query_stmts if '_Target' in graph_mix.stmts[stmt_id].raw_label}

    query_stmt_tups = {(stmt.raw_label, stmt.head_id, stmt.tail_id) for stmt in [graph_mix.stmts[item] for item in query_stmts if graph_mix.stmts[item].tail_id]}
    die_victim_list = {stmt.tail_id for stmt in [graph_mix.stmts[item] for item in query_stmts if graph_mix.stmts[item].tail_id] if 'Die_Victim' in stmt.raw_label}

    # Account for EREs which have both 'Attacker' and 'Target' statements in the query set
    need_atk_set = query_events_w_trg - query_events_w_atk
    need_trg_set = query_events_w_atk - query_events_w_trg

    # Filter out EREs which have no 'Attacker' or 'Target' statements in the graph salad
    need_atk_set = {ere_id for ere_id in need_atk_set if ('_Attacker' in '.'.join([graph_mix.stmts[stmt_id].raw_label for stmt_id in graph_mix.eres[ere_id].stmt_ids]))}
    need_trg_set = {ere_id for ere_id in need_trg_set if ('_Target' in '.'.join([graph_mix.stmts[stmt_id].raw_label for stmt_id in graph_mix.eres[ere_id].stmt_ids]))}

    # Test stmts one after another for their validity
    for index in sorted_pred_indices:
        cand_stmt_id = batch[0]['stmt_mat_ind'].get_word(batch[0]['candidates'][index])
        cand_stmt = graph_mix.stmts[cand_stmt_id]

        # If either need_atk_set or need_trg_set are non-empty, we know there is a candidate statement which is guaranteed to pass the two tests in the "else" branch
        if need_atk_set or need_trg_set:
            if cand_stmt.head_id in need_atk_set and '_Attacker' in cand_stmt.raw_label:
                return index
            elif cand_stmt.head_id in need_trg_set and '_Target' in cand_stmt.raw_label:
                return index
            else:
                # If either need_atk_set or need_trg_set are non-empty, we know there is a candidate statement which is guaranteed to pass the below two tests
                continue
        else:
            if (cand_stmt.raw_label, cand_stmt.head_id, cand_stmt.tail_id) in query_stmt_tups:
                continue
            if 'Die_Victim' in cand_stmt.raw_label and cand_stmt.tail_id in die_victim_list:
                continue
            return index
    # If all stmts are invalid, just return the first index (in general this case shouldn't happen)
    return sorted_pred_indices[0]


def remove_duplicate_events(batch):
    """Remove events that agree in event type and in the IDs of all arguments"""
    graph_mix = batch[0]['graph_mix']
    query_stmts = {batch[0]['stmt_mat_ind'].get_word(item) for item in batch[0]['query_stmts']}
    query_eres = {batch[0]['ere_mat_ind'].get_word(item) for item in batch[0]['query_eres']}
    tups = set()
    # For all query ERE objects that are classified as an "Event"/"Relation"
    for ere in [graph_mix.eres[item] for item in query_eres if graph_mix.eres[item].category in ['Event', 'Relation']]:
        # Determine the set of all event typing labels (as determined by the labels of all surrounding non-typing statements); with our synthetic data, we should have only one of these,
        # so we assume it's a singleton set and take the single element
        # This would be, for example, 'Life.Die'
        event_type_label = list({graph_mix.stmts[stmt_id].raw_label for stmt_id in ere.stmt_ids if not graph_mix.stmts[stmt_id].tail_id})[0]
        # Find the set of role/non-typing statements which are attached to the current ERE and appear in the query set
        ere_query_stmt_ids = set.intersection(query_stmts, {stmt_id for stmt_id in ere.stmt_ids if graph_mix.stmts[stmt_id].tail_id})
        role_stmt_tups = set()
        # Find all non-typing (or "role") statements for the current ERE (a second time); record their label, head ERE, and tail ERE
        for stmt in [graph_mix.stmts[stmt_id] for stmt_id in ere_query_stmt_ids]:
            role_stmt_tups.add((stmt.raw_label, stmt.head_id, stmt.tail_id))
        tup = (event_type_label, frozenset(role_stmt_tups))
        # Remove any query statements that belong to an offending/duplicate ERE
        if tup in tups:
            query_stmts -= ere.stmt_ids
        else:
            tups.add(tup)
    batch[0]['query_stmts'] = np.asarray([batch[0]['stmt_mat_ind'].get_index(item, add=False) for item in query_stmts])