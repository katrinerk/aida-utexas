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
        candidate_ids += [stmt_mat_ind.get_index(neigh_stmt, add=False) for neigh_stmt in neigh_stmts if stmt_mat_ind.get_index(neigh_stmt, add=False) not in query_stmts]

    candidate_ids = list(set(candidate_ids))

    # Determine the appropriate class label for each candidate statement
    [stmt_class_labels.append(1 if graph_mix.stmts[stmt_mat_ind.get_word(candidate_id)].graph_id == target_graph_id else 0) for candidate_id in candidate_ids]

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

    random_correct_indices = random.choice(correct_indices)
    return random_correct_indices

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
    graph_dict['query_stmts'] = np.insert(graph_dict['query_stmts'], len(graph_dict['query_stmts']), graph_dict['candidates'][selected_index])

    cand_stmt = graph_dict['graph_mix'].stmts[graph_dict['stmt_mat_ind'].get_word(graph_dict['candidates'][selected_index])]

    # Record the head/tail ERE of the admitted stmt for later use
    temp = [cand_stmt.head_id, cand_stmt.tail_id]

    if not temp[1]:
        temp = temp[0:1]

    sel_val = graph_dict['candidates'][selected_index]

    # Delete the selected candidate list element and its corresponding entry in the list of label inputs
    del graph_dict['candidates'][selected_index]

    if len(graph_dict['stmt_class_labels']) != 0:##TO-DO
        del graph_dict['stmt_class_labels'][selected_index]

    # Determine the set of EREs (if any) introduced by the candidate statement to the query set
    set_diff = set([graph_dict['ere_mat_ind'].get_index(item, add=False) for item in temp]) - set(graph_dict['query_eres'])

    if set_diff:
        # For each new ERE in the query set, add all surrounding statements (less existing query statements) to the candidate set
        for ere_iter in set_diff:
            stmt_neighs = torch.cat((torch.nonzero(torch.from_numpy(graph_dict['adj_head'][ere_iter]), as_tuple=False),
                                     torch.nonzero(torch.from_numpy(graph_dict['adj_tail'][ere_iter]), as_tuple=False),
                                     torch.nonzero(torch.from_numpy(graph_dict['adj_type'][ere_iter]), as_tuple=False)), dim=0).reshape((-1))
            stmt_neighs = [stmt_neigh for stmt_neigh in stmt_neighs[stmt_neighs != sel_val].tolist() if stmt_neigh not in graph_dict['candidates']]
            graph_dict['candidates'] += stmt_neighs

            graph_dict['stmt_class_labels'] = graph_dict['stmt_class_labels'] + [1 if graph_dict['graph_mix'].stmts[graph_dict['stmt_mat_ind'].get_word(stmt_iter)].graph_id == graph_dict['target_graph_id']
                                                                         else 0 for stmt_iter in stmt_neighs]

    graph_dict['query_eres'] = set.union(graph_dict['query_eres'], set_diff)

def select_valid_hypothesis(graph_dict, prediction):
    """Select the index of the highest-scoring valid candidate stmt"""
    graph_mix = graph_dict['graph_mix']
    # get the indices that sort the prediction tensor
    sorted_pred_indices = torch.argsort(prediction, descending=True)
    query_stmts = {graph_dict['stmt_mat_ind'].get_word(item) for item in graph_dict['query_stmts']}

    # Test stmts one after another for their validity
    for index in sorted_pred_indices:
        cand_stmt = graph_mix.stmts[graph_dict['stmt_mat_ind'].get_word(graph_dict['candidates'][index])]
        query_stmts_arguments = {(stmt.raw_label, stmt.head_id, stmt.tail_id) for stmt in [graph_mix.stmts[item] for item in query_stmts]}
        # Remove stmts that agree in stmt type and in the IDs of all arguments
        if (cand_stmt.raw_label, cand_stmt.head_id, cand_stmt.tail_id) in query_stmts_arguments:
            continue
        # A person is not supposed to die twice
        if cand_stmt.raw_label == 'Life.Die_Victim' and cand_stmt.tail_id in [stmt.tail_id for stmt in [graph_mix.stmts[item] for item in query_stmts] if stmt.raw_label == 'Life.Die_Victim']:
            continue
        # If the stmt is valid, just return its index
        # Since we already sorted the prediction tensor, the score of the stmt is highest among all valid candidate stmts 
        return index
    # If all stmts are invalid, just return the first index (in general this case shouldn't happen)
    return sorted_pred_indices[0]

def remove_duplicate_events(graph_dict):
    """Remove events that agree in event type and in the IDs of all arguments"""
    graph_mix = graph_dict['graph_mix']
    query_stmts = {graph_dict['stmt_mat_ind'].get_word(item) for item in graph_dict['query_stmts']}
    query_eres = {graph_dict['ere_mat_ind'].get_word(item) for item in graph_dict['query_eres']}
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
    graph_dict['query_stmts'] = np.asarray([graph_dict['stmt_mat_ind'].get_index(item, add=False) for item in query_stmts])
    
