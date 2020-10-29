# Original author: Su Wang, 2019
# Modified by Alex Tomkovich in 2019/2020

######
# This file contains various utilities for sampling graph salads from a data set,
# updating graph salad inference states, and determining loss during training.
######

from copy import deepcopy
import dill
import numpy as np
import random
import os
import re
import torch


# Get the set of candidate stmts available for evaluation at the current time step;
# the set of candidate stmts consist of all stmts which are attached to an ERE
# already captured by the query set
def get_candidates(graph_dict):
    query_stmts = graph_dict['query_stmts']
    query_eres = graph_dict['query_eres']
    stmt_mat_ind = graph_dict['stmt_mat_ind']
    ere_mat_ind = graph_dict['ere_mat_ind']
    graph_mix = graph_dict['graph_mix']
    candidate_ids = []
    stmt_class_labels = []

    target_graph_id = graph_dict['target_graph_id']

    for ere_iter in query_eres:
        neigh_stmts = graph_mix.eres[ere_mat_ind.get_word(ere_iter)].stmt_ids
        candidate_ids += [stmt_mat_ind.get_index(neigh_stmt, add=False) for neigh_stmt in neigh_stmts if
                          stmt_mat_ind.get_index(neigh_stmt, add=False) not in query_stmts]

    candidate_ids = list(set(candidate_ids))

    # Determine the appropriate class label for each candidate statement
    [stmt_class_labels.append(
        1 if graph_mix.stmts[stmt_mat_ind.get_word(candidate_id)].graph_id == target_graph_id else 0) for candidate_id
     in candidate_ids]

    return candidate_ids, stmt_class_labels


class DataIterator:
    def __init__(self, data_dir):
        self.file_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]
        self.cursor = 0
        self.epoch = 0
        self.shuffle()
        self.size = len(self.file_paths)

    def shuffle(self):
        self.file_paths = sorted(self.file_paths)
        random.seed(self.epoch)
        random.shuffle(self.file_paths)

    def next_batch(self):
        if self.cursor >= self.size:
            self.cursor = 0
            self.epoch += 1

        # Data comes in the form of a "graph_dict" dictionary which contains the following key-value pairs:
        # 'graph_mix'    -     --> pickled graph salad object
        # 'ere_mat_ind'        --> Indexer object which maps ERE IDs to indices in an adjacency matrix
        # 'stmt_mat_ind'       --> Indexer object which maps stmt IDs to indices in an adjacency matrix
        # 'adj_head'           --> (num_eres x num_stmts) adjacency matrix;
        #                          contains a 1 in position (x, y) if statement y is attached to ERE x at the head/subject (does NOT include typing statements)
        # 'adj_tail'           --> same as above, but for statements attached to EREs at the tail
        # 'adj_type'           --> same as above, but for typing statements attached to EREs
        # 'ere_labels'         --> lists of indices for labels associated with EREs
        # 'stmt_labels'        --> lists of indices for labels associated with stmts
        # 'num_word2vec_ere'   --> number of ERE names identified in the Word2Vec vocabulary used
        # 'num_word2vec_stmts' --> number of stmt labels identified in the Word2Vec vocabulary used

        graph_dict = dill.load(open(self.file_paths[self.cursor], "rb"))

        # Determine the initial list of candidate statements (and their class labels)
        candidate_ids_list, stmt_class_labels_list = get_candidates(graph_dict)

        self.cursor += 1

        graph_dict['candidates'] = candidate_ids_list
        graph_dict['stmt_class_labels'] = stmt_class_labels_list

        return graph_dict


# Get a random target-graph statement from the available pool of candidates;
# if "use_highest_ranked_gold," select the target-graph stmt which the
# model ranks most highly; otherwise, select a random target-graph stmt
def get_random_gold_label(graph_dict, prediction, use_high_ranked_gold):
    correct_indices = [index for index, label in enumerate(graph_dict['stmt_class_labels']) if label == 1]

    if use_high_ranked_gold:
        correct_indices = [correct_indices[int(torch.argmax(prediction[correct_indices]))]]

    random_correct_index = random.sample(correct_indices, 1)[0]
    return random_correct_index


# Convert the list of candidate class labels into a tensor
def get_tensor_labels(graph_dict, device=torch.device("cpu")):
    return torch.FloatTensor(np.array(deepcopy(graph_dict['stmt_class_labels']))).to(device)


# Multi-correct NLL loss function after Durrett and Klein, 2013
def multi_correct_nll_loss(predictions, trues, device=torch.device("cpu")):
    nll = torch.Tensor([]).to(device)

    for prediction, true in zip(predictions, trues):
        masked_log_probability = torch.mul(prediction, true)
        log_probabilities = masked_log_probability[torch.where(masked_log_probability != 0)[0].squeeze()]
        if len(log_probabilities.shape) == 0:
            log_probabilities = log_probabilities.unsqueeze(0)
        log_sum_probability = (-1 * torch.logsumexp(log_probabilities, dim=0)).unsqueeze(0)
        if log_sum_probability == float("inf"):
            continue
        nll = torch.cat([nll, log_sum_probability])

    if len(nll) == 0:
        nll = torch.Tensor([0]).to(device)
    return nll.mean()


# Given the index of the admitted stmt, update the inference state
def next_state(graph_dict, selected_index):
    # Add the selected candidate to the query set
    graph_dict['query_stmts'] = np.insert(graph_dict['query_stmts'], len(graph_dict['query_stmts']),
                                          graph_dict['candidates'][selected_index])

    cand_stmt = graph_dict['graph_mix'].stmts[
        graph_dict['stmt_mat_ind'].get_word(graph_dict['candidates'][selected_index])]

    # Record the head/tail ERE of the admitted stmt for later use
    temp = [cand_stmt.head_id, cand_stmt.tail_id]

    if not temp[1]:
        temp = temp[0:1]

    sel_val = graph_dict['candidates'][selected_index]

    # Delete the selected candidate list element and its corresponding entry in the list of label inputs
    del graph_dict['candidates'][selected_index]

    if len(graph_dict['stmt_class_labels']) != 0:  ##TO-DO
        del graph_dict['stmt_class_labels'][selected_index]

    # Determine the set of EREs (if any) introduced by the candidate statement to the query set
    set_diff = {graph_dict['ere_mat_ind'].get_index(item, add=False) for item in temp} - set(graph_dict['query_eres'])

    if set_diff:
        # For each new ERE in the query set, add all surrounding statements (less existing query statements) to the candidate set
        for ere_iter in set_diff:
            stmt_neighs = torch.cat((torch.nonzero(torch.from_numpy(graph_dict['adj_head'][ere_iter]), as_tuple=False),
                                     torch.nonzero(torch.from_numpy(graph_dict['adj_tail'][ere_iter]), as_tuple=False),
                                     torch.nonzero(torch.from_numpy(graph_dict['adj_type'][ere_iter]), as_tuple=False)),
                                    dim=0).reshape((-1))
            stmt_neighs = [stmt_neigh for stmt_neigh in stmt_neighs[stmt_neighs != sel_val].tolist() if
                           stmt_neigh not in graph_dict['candidates']]
            graph_dict['candidates'] += stmt_neighs

            if len(graph_dict['stmt_class_labels']) != 0:
                graph_dict['stmt_class_labels'] = graph_dict['stmt_class_labels'] + [
                    1 if graph_dict['graph_mix'].stmts[graph_dict['stmt_mat_ind'].get_word(stmt_iter)].graph_id ==
                         graph_dict['target_graph_id']
                    else 0 for stmt_iter in stmt_neighs]

    graph_dict['query_eres'] = set.union(graph_dict['query_eres'], set_diff)


def select_valid_hypothesis(graph_dict, prediction):
    """Select the index of the highest-scoring valid candidate stmt"""
    graph_mix = graph_dict['graph_mix']
    # get the indices that sort the prediction tensor
    sorted_pred_indices = torch.argsort(prediction, descending=True)
    query_stmts = {graph_dict['stmt_mat_ind'].get_word(item) for item in graph_dict['query_stmts']}

    # Find all event EREs with 'Attacker' statements in the query set
    query_events_w_atk = {graph_mix.stmts[stmt_id].head_id for stmt_id in query_stmts if
                          '_Attacker' in graph_mix.stmts[stmt_id].raw_label}
    # Find all event EREs with 'Target' statements in the query set
    query_events_w_trg = {graph_mix.stmts[stmt_id].head_id for stmt_id in query_stmts if
                          '_Target' in graph_mix.stmts[stmt_id].raw_label}

    # Find all event EREs with 'Killer' statements in the query set
    query_events_w_kle = {graph_mix.stmts[stmt_id].head_id for stmt_id in query_stmts if
                          '_Killer' in graph_mix.stmts[stmt_id].raw_label}
    # Find all event EREs with 'Life.Die' and '_Victim' statements in the query set
    query_events_w_vct = {graph_mix.stmts[stmt_id].head_id for stmt_id in query_stmts if
                          all([x in graph_mix.stmts[stmt_id].raw_label for x in ['Life.Die', '_Victim']])}

    query_stmt_tups = {(stmt.raw_label, stmt.head_id, stmt.tail_id) for stmt in
                       [graph_mix.stmts[item] for item in query_stmts if graph_mix.stmts[item].tail_id]}
    die_victim_list = {stmt.tail_id for stmt in
                       [graph_mix.stmts[item] for item in query_stmts if graph_mix.stmts[item].tail_id] if
                       all([x in stmt.raw_label for x in ['Life.Die', '_Victim']])}

    # Account for EREs which have both 'Attacker' and 'Target' statements in the query set
    need_atk_set = query_events_w_trg - query_events_w_atk
    need_trg_set = query_events_w_atk - query_events_w_trg

    # Account for EREs which have both 'Die_Killer' and 'Life.Die'/'_Victim' statements in the query set
    need_kle_set = query_events_w_vct - query_events_w_kle
    need_vct_set = query_events_w_kle - query_events_w_vct

    # Filter out EREs which have no 'Attacker' or 'Target' statements in the graph salad
    need_atk_set = {ere_id for ere_id in need_atk_set if ('_Attacker' in '.'.join(
        [graph_mix.stmts[stmt_id].raw_label for stmt_id in graph_mix.eres[ere_id].stmt_ids]))}
    need_trg_set = {ere_id for ere_id in need_trg_set if ('_Target' in '.'.join(
        [graph_mix.stmts[stmt_id].raw_label for stmt_id in graph_mix.eres[ere_id].stmt_ids]))}

    # Filter out EREs which have no 'Killer' or 'Life.Die'/'_Victim' statements in the graph salad
    need_kle_set = {ere_id for ere_id in need_kle_set if ('_Killer' in '.'.join(
        [graph_mix.stmts[stmt_id].raw_label for stmt_id in graph_mix.eres[ere_id].stmt_ids]))}
    need_vct_set = {ere_id for ere_id in need_vct_set if
                    len([stmt_id for stmt_id in graph_mix.eres[ere_id].stmt_ids if all([x in graph_mix.stmts[stmt_id].raw_label for x in ['Life.Die', '_Victim']])]) > 0}

    # Test stmts one after another for their validity
    for index in sorted_pred_indices:
        cand_stmt_id = graph_dict['stmt_mat_ind'].get_word(graph_dict['candidates'][index])
        cand_stmt = graph_mix.stmts[cand_stmt_id]

        attk_and_targ = False

        if '_Attacker' in cand_stmt.raw_label:
            for stmt_id in set.intersection(query_stmts, graph_mix.eres[cand_stmt.head_id].stmt_ids):
                if '_Target' in graph_mix.stmts[stmt_id].raw_label and cand_stmt.tail_id == graph_mix.stmts[stmt_id].tail_id:
                    attk_and_targ = True
                    break
        if '_Target' in cand_stmt.raw_label:
            for stmt_id in set.intersection(query_stmts, graph_mix.eres[cand_stmt.head_id].stmt_ids):
                if '_Attacker' in graph_mix.stmts[stmt_id].raw_label and cand_stmt.tail_id == graph_mix.stmts[stmt_id].tail_id:
                    attk_and_targ = True
                    break

        kllr_and_vict = False

        if '_Killer' in cand_stmt.raw_label:
            for stmt_id in set.intersection(query_stmts, graph_mix.eres[cand_stmt.head_id].stmt_ids):
                if all([x in graph_mix.stmts[stmt_id].raw_label for x in ['Life.Die', '_Victim']]) and cand_stmt.tail_id == graph_mix.stmts[stmt_id].tail_id:
                    kllr_and_vict = True
                    break
        if all([x in cand_stmt.raw_label for x in ['Life.Die', '_Victim']]):
            for stmt_id in set.intersection(query_stmts, graph_mix.eres[cand_stmt.head_id].stmt_ids):
                if '_Killer' in graph_mix.stmts[stmt_id].raw_label and cand_stmt.tail_id == graph_mix.stmts[stmt_id].tail_id:
                    kllr_and_vict = True
                    break

        # If at least one 'need_set' is non-empty, we know there is a candidate statement which is guaranteed to pass the two tests in the "else" branch
        if any([need_atk_set, need_trg_set, need_kle_set, need_vct_set]):
            if need_atk_set or need_trg_set:
                if cand_stmt.head_id in need_atk_set and '_Attacker' in cand_stmt.raw_label and not attk_and_targ:
                    return index
                elif cand_stmt.head_id in need_trg_set and '_Target' in cand_stmt.raw_label and not attk_and_targ:
                    return index
            elif need_kle_set or need_vct_set:
                if cand_stmt.head_id in need_kle_set and '_Killer' in cand_stmt.raw_label and not kllr_and_vict:
                    return index
                elif cand_stmt.head_id in need_vct_set and all([x in cand_stmt.raw_label for x in ['Life.Die', '_Victim']]) and not kllr_and_vict:
                    if cand_stmt.tail_id not in die_victim_list:
                        return index
        else:
            if (cand_stmt.raw_label, cand_stmt.head_id, cand_stmt.tail_id) in query_stmt_tups:
                continue
            if all([x in cand_stmt.raw_label for x in ['Life.Die', '_Victim']]) and cand_stmt.tail_id in die_victim_list:
                continue
            if (not attk_and_targ) and (not kllr_and_vict):
                return index

    # If all stmts are invalid, just return the first index (in general this case shouldn't happen)
    return sorted_pred_indices[0]


def remove_duplicate_events(graph_dict):
    """Remove events that agree in event type and in the IDs of all arguments"""
    graph_mix = graph_dict['graph_mix']
    query_stmts = {graph_dict['stmt_mat_ind'].get_word(item) for item in graph_dict['query_stmts']}
    query_eres = {graph_dict['ere_mat_ind'].get_word(item) for item in graph_dict['query_eres']}
    tups = []

    # For all query ERE objects that are classified as an "Event"/"Relation"
    for ere_id in [item for item in query_eres if graph_mix.eres[item].category in ['Event', 'Relation']]:
        ere = graph_mix.eres[ere_id]

        # Determine the set of all event typing labels (as determined by the labels of all surrounding non-typing statements); with our synthetic data, we should have only one of these,
        # so we assume it's a singleton set and take the single element
        # This would be, for example, 'Life.Die'
        event_type_labels = {graph_mix.stmts[stmt_id].raw_label for stmt_id in ere.stmt_ids if
                             not graph_mix.stmts[stmt_id].tail_id}
        # Find the set of role/non-typing statements which are attached to the current ERE and appear in the query set
        ere_query_stmt_ids = set.intersection(query_stmts,
                                              {stmt_id for stmt_id in ere.stmt_ids if graph_mix.stmts[stmt_id].tail_id})
        role_stmt_tups = set()
        # Find all non-typing (or "role") statements for the current ERE (a second time); record their label, head ERE, and tail ERE
        for stmt in [graph_mix.stmts[stmt_id] for stmt_id in ere_query_stmt_ids]:
            role_stmt_tups.add((stmt.raw_label, stmt.head_id, stmt.tail_id))
        tup = (event_type_labels, role_stmt_tups)
        # Remove any query statements that belong to an offending/duplicate ERE
        dup = False

        for item in tups:
            if set.intersection(tup[0], item[0]) and tup[1] == item[1]:
                dup = True
                break

        if dup:
            query_stmts -= ere.stmt_ids
            query_eres.discard(ere_id)
        else:
            tups.append(tup)

    graph_dict['query_stmts'] = np.asarray([graph_dict['stmt_mat_ind'].get_index(item, add=False) for item in query_stmts])
    graph_dict['query_eres'] = np.asarray([graph_dict['ere_mat_ind'].get_index(item, add=False) for item in query_eres])

def remove_place_only_events(graph_dict):
    graph_mix = graph_dict['graph_mix']
    query_stmts = {graph_dict['stmt_mat_ind'].get_word(item) for item in graph_dict['query_stmts']}
    query_eres = {graph_dict['ere_mat_ind'].get_word(item) for item in graph_dict['query_eres']}

    for ere_id in [item for item in query_eres if graph_mix.eres[item].category == 'Event']:
        ere = graph_mix.eres[ere_id]

        role_stmts = set.intersection(query_stmts, {stmt_id for stmt_id in ere.stmt_ids if graph_mix.stmts[stmt_id].tail_id})

        place_role_stmts = {stmt_id for stmt_id in role_stmts if '_Place' in graph_mix.stmts[stmt_id].raw_label}

        # Even if both of these are 0, we should still delete the event; we don't want events with no arguments
        if len(place_role_stmts) == len(role_stmts):
            query_stmts -= ere.stmt_ids
            query_eres.discard(ere_id)

    graph_dict['query_stmts'] = np.asarray([graph_dict['stmt_mat_ind'].get_index(item, add=False) for item in query_stmts])
    graph_dict['query_eres'] = np.asarray([graph_dict['ere_mat_ind'].get_index(item, add=False) for item in query_eres])
