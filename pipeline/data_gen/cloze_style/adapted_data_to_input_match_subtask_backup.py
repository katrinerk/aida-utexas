""" Json, JsonJust and original text (i.e. rsd.txt files) to model inputs (i.e. document mixtures).

Author: Su Wang; 2019.
Usage:
python3 data_to_input.py \
    --<OPTION1>=... \
    --<OPTION2>=... \
    ...

"""

import argparse
import json
import os
import random
import re
import time
import math
import sys
from collections import defaultdict
from copy import deepcopy
from operator import itemgetter
from pathlib import Path

import dill
from tqdm import tqdm
import itertools
import numpy as np
import h5py
from elmo_tokenize import check_poss_stmt_span, get_rej_ere_ids, check_spans_distinct, check_poss_ere_stmt_spans, check_distinct_ere_pair, get_stmt_spans

##### GENERAL #####

class Indexer(object):
    """Word to index bidirectional mapping."""

    def __init__(self, start_symbol="<s>", end_symbol="</s>"):
        """Initializing dictionaries and (hard coded) special symbols."""
        self.word_to_index = {}
        self.index_to_word = {}
        self.size = 0
        # Hard-code special symbols.
        self.get_index("PAD", add=True)
        self.get_index("UNK", add=True)
        self.get_index(start_symbol, add=True)
        self.get_index(end_symbol, add=True)

    def __repr__(self):
        """Print size info."""
        return "This indexer currently has %d words" % self.size

    def get_word(self, index):
        """Get word by index if its in range. Otherwise return `UNK`."""
        return self.index_to_word[index] if index < self.size and index >= 0 else "UNK"

    def get_index(self, word, add):
        """Get index by word. If `add` is on, also append word to the dictionaries."""
        if self.contains(word):
            return self.word_to_index[word]
        elif add:
            self.word_to_index[word] = self.size
            self.index_to_word[self.size] = word
            self.size += 1
            return self.word_to_index[word]
        return self.word_to_index["UNK"]

    def contains(self, word):
        """Return True/False to indicate whether a word is in the dictionaries."""
        return word in self.word_to_index

    def add_sentence(self, sentence, add):
        """Add all the words in a sentence (a string) to the dictionary."""
        indices = [self.get_index(word, add) for word in sentence.split()]
        return indices

    def add_document(self, document_path, add):
        """Add all the words in a document (a path to a text file) to the dictionary."""
        indices_list = []
        with open(document_path, "r") as document:
            for line in document:
                indices = self.add_sentence(line, add)
                indices_list.append(indices)
        return indices_list

    def to_words(self, indices):
        """Indices (ints) -> words (strings) conversion."""
        return [self.get_word(index) for index in indices]

    def to_sent(self, indices):
        """Indices (ints) -> sentence (1 string) conversion."""
        return " ".join(self.to_words(indices))

    def to_indices(self, words):
        """Words (strings) -> indices (ints) conversion."""
        return [self.get_index(word, add=False) for word in words]


##### DATA STRUCTURE #####

class Ere:
    """Node object."""

    def __init__(self,
                 graph_id, category,
                 ere_id, label):
        """Initializer.

        Args:
            graph_id: unique id of a graph, e.g. enwiki-20100312_0000494023.
            category: Event | Relation | Entity.
            node_id: unique id of the node entry, .e.g. http://www.isi.edu/gaia/....
            label: node name string, e.g. Ohio.
            stmt_ids: indices of stmts, for AIDA output.
        """
        self.graph_id = graph_id
        self.category = category
        self.id = ere_id
        self.label = label
        self.neighbor_ere_ids = set()
        self.stmt_ids = set()

    def __repr__(self):
        return "[ERE] | ID: %s | Label: %s" % (self.id, self.label)

    @staticmethod
    def entry_type():
        return "Ere"


class Stmt:

    def __init__(self,
                 graph_id,
                 stmt_id, raw_label, label,
                 head_id, tail_id):
        """Initializer.

        Args:
            graph_id: unique id of a graph, e.g. enwiki-20100312_0000494023.
            edge_id: unique id of the node entry, e.g. ub1bL181C1.
            label: edge name string, e.g. Conflict.Attack_Attacker.
            head_id: unique id of the head node of the edge.
            tail_id: .. of the tail node of the edge.
        """
        self.graph_id = graph_id
        self.id = stmt_id
        self.dup_ids = set()
        self.raw_label = raw_label
        self.label = label
        self.head_id = head_id
        self.tail_id = tail_id

    def __repr__(self):
        return "[STMT] | ID: %s | Label: %s" % (self.id, self.label)

    def link_info(self):
        return "Head: %s \nTail: %s" % (self.head_id, self.tail_id)

    @staticmethod
    def entry_type():
        return "Stmt"


class Graph:
    """Graph wrap for nodes and edges."""

    def __init__(self, graph_id):
        """Initializer.

        Args:
            graph_id: unique id of a graph, e.g. enwiki-20100312_0000494023.
        """
        self.graph_id = graph_id
        self.eres = dict()  # a node dictionary.
        self.stmts = dict()  # an edge dictionary.
        # the number of neighbors reachable from 1 step of each node
        self.connectedness_one_step = None
        # the number of neighbors reachable from 2 steps of each node
        self.connectedness_two_step = None
        # the mapping from attributes (types) to node ids
        self.attribute_to_ere_ids = None

    def __repr__(self):
        return "[GRAPH] | ID: %s | #N: %d | #E: %d" % (self.graph_id, len(self.eres), len(self.stmts))

    def is_empty(self):
        return len(self.eres) == 0 and len(self.stmts) == 0

    def unique_id(self, entry_id):
        return self.graph_id + '_' + entry_id


def load_eres(graph, graph_js):
    for entry_id, entry in graph_js.items():
        # Ignore all Statements, SameAsClusters, and ClusterMemberships first
        if entry["type"] in ["Statement", "SameAsCluster", "ClusterMembership"]:
            continue

        ere_id = graph.unique_id(entry_id)

        if entry["type"] == "Relation":  # relation nodes have no labels.
            graph.eres[ere_id] = Ere(
                graph_id=graph.graph_id,
                category=entry["type"],
                ere_id=ere_id,
                label=["Relation"]
            )
        else:  # event or entity, same processing procedure.
            keep_labels = []

            if "name" in entry.keys():
                for label in entry["name"]:
                    match = re.search('[_ ]+', label)

                    if match and match.span()[1] - match.span()[0] == len(label):
                        continue

                    keep_labels.append(label)

            graph.eres[ere_id] = Ere(
                graph_id=graph.graph_id,
                category=entry["type"],
                ere_id=ere_id,
                label=keep_labels
            )


def load_statements(graph, graph_js):
    """Load edges. All are Stmts, including attributes and links."""
    seen_stmts = dict()

    for entry_id, entry in graph_js.items():
        if entry["type"] != "Statement":
            continue

        stmt_id = graph.unique_id(entry_id)
        subj_id = graph.unique_id(entry['subject'])

        if entry["predicate"] == "type":  # typing statements
            type_id = entry["object"]
            type_label = entry["object"].split("#")[-1]

            tup = ('Type', subj_id, type_id)

            if tup in seen_stmts.keys():
                graph.stmts[seen_stmts[tup]].dup_ids.add(stmt_id)
                continue

            seen_stmts[tup] = stmt_id

            graph.stmts[stmt_id] = Stmt(
                graph_id=graph.graph_id,
                stmt_id=stmt_id,
                raw_label=type_label,
                label=re.sub("[.]", " ", type_label).split(),
                head_id=subj_id,
                tail_id=None
            )

            graph.eres[subj_id].stmt_ids.add(stmt_id)
        else:  # edge statements
            obj_id = graph.unique_id(entry['object'])

            split_label = re.sub("[._]", " ", entry["predicate"]).split()

            tup = (' '.join(split_label), subj_id, obj_id)

            if tup in seen_stmts.keys():
                graph.stmts[seen_stmts[tup]].dup_ids.add(stmt_id)
                continue

            seen_stmts[tup] = stmt_id

            graph.stmts[stmt_id] = Stmt(
                graph_id=graph.graph_id,
                stmt_id=stmt_id,
                raw_label=entry["predicate"],
                label=split_label,
                head_id=subj_id,
                tail_id=obj_id)

            graph.eres[subj_id].neighbor_ere_ids.add(obj_id)
            graph.eres[obj_id].neighbor_ere_ids.add(subj_id)

            graph.eres[subj_id].stmt_ids.add(stmt_id)
            graph.eres[obj_id].stmt_ids.add(stmt_id)

def remove_singletons(graph):
    """Remove singletons. NB: a few graphs may be empty after this op."""
    singleton_ids = [ere_id for ere_id in graph.eres.keys() if len(graph.eres[ere_id].neighbor_ere_ids) == 0]

    for singleton_id in singleton_ids:
        for stmt_id in graph.eres[singleton_id].stmt_ids:
            del graph.stmts[stmt_id]
        del graph.eres[singleton_id]


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


def build_attribute_mapping(graph):
    """Get attribute information from a graph, update in an attribute -> node id dictionary."""
    attribute_to_ere_ids = defaultdict(set)
    for ere_id, ere in graph.eres.items():
        if ere.category == 'Relation':
            continue
        elif ere.category == 'Event':
            role_stmt_ids = [stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id]
            if len(role_stmt_ids) < 1:
                continue
        elif ere.category == 'Entity':
            role_stmt_ids = [stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if (graph.stmts[stmt_id].tail_id == ere_id) and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Event')]

            if len(role_stmt_ids) < 1:
                continue
        else:
            print('Should not be here: unexpected node category', ere.category)

        type_stmt_ids = [stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if (graph.stmts[stmt_id].tail_id is None)]

        for type_stmt_id in type_stmt_ids:
            attribute = (' ').join(graph.stmts[type_stmt_id].label)
            attribute_to_ere_ids[attribute].add(ere_id)

    graph.attribute_to_ere_ids = {}

    for attribute, ere_ids in attribute_to_ere_ids.items():
        ere_ids_with_connectedness = [(ere_id, graph.connectedness_two_step[ere_id], graph.connectedness_one_step[ere_id]) for ere_id in ere_ids]

        ere_ids_with_connectedness = sorted(ere_ids_with_connectedness, key=itemgetter(1, 2), reverse=True)

        graph.attribute_to_ere_ids[attribute] = ere_ids_with_connectedness


def read_graph(graph_id, graph_js):
    graph = Graph(graph_id)

    load_eres(graph, graph_js)
    load_statements(graph, graph_js)

    remove_singletons(graph)

    compute_connectedness(graph)
    build_attribute_mapping(graph)

    return graph


##### GRAPH LOADER & MIXER #####

def verify_dir(dir):
    """Make and dir if one doesn't exist."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_random_entries(first_graph_list, second_graph_list, num_sources, used_pairs):
    if second_graph_list is None:
        graphs = random.sample(first_graph_list, num_sources)
        while (len(set([graph.graph_id for graph in graphs])) < len(graphs)) or (set([graph.graph_id for graph in graphs]) in used_pairs):
            graphs = random.sample(first_graph_list, num_sources)
    else:
        graphs = [random.choice(first_graph_list), random.choice(second_graph_list)] + [random.choice(first_graph_list + second_graph_list) for _ in range(num_sources - 2)]
        while (len(set([graph.graph_id for graph in graphs])) < len(graphs)) or (set([graph.graph_id for graph in graphs]) in used_pairs):
            graphs = [random.choice(first_graph_list), random.choice(second_graph_list)] + [random.choice(first_graph_list + second_graph_list) for _ in range(num_sources - 2)]

    return [deepcopy(graph) for graph in graphs]


def retrieve_related_stmt_ids(graph, root_id, max_num_neigh_ere_ids=2):
    root_ere = graph.eres[root_id]

    # Select at most 2 neighbor ERE's
    neighbor_ere_ids = root_ere.neighbor_ere_ids
    if len(neighbor_ere_ids) > max_num_neigh_ere_ids:
        neighbor_ere_ids = random.sample(neighbor_ere_ids, max_num_neigh_ere_ids)

    stmt_set = set()

    for neighbor_ere_id in neighbor_ere_ids:
        stmt_set.update([stmt_id for stmt_id in graph.eres[root_id].stmt_ids if (neighbor_ere_id in [graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id]) or (graph.stmts[stmt_id].tail_id is None)])
        stmt_set.update([stmt_id for stmt_id in graph.eres[neighbor_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id is None])
    return stmt_set

def select_shared_ere_ids(graphs, num_shared_eres, min_connectedness_one_step, min_connectedness_two_step):
    shared_ere_types = set.intersection(*[set(graph.attribute_to_ere_ids.keys()) for graph in graphs])
    if len(shared_ere_types) == 0:
        return None

    shared_ere_type_ids_list = []

    for ere_type in shared_ere_types:
        for ere_ids_with_connectedness in itertools.product(*[graph.attribute_to_ere_ids[ere_type] for graph in graphs]):
            ere_ids, ere_connectedness_scores_two_step, ere_connectedness_scores_one_step = zip(*ere_ids_with_connectedness)

            one_step_min = min(ere_connectedness_scores_one_step)
            if one_step_min < min_connectedness_one_step:
                continue

            two_step_min = min(ere_connectedness_scores_two_step)
            if two_step_min < min_connectedness_two_step:
                continue

            one_step_sum = sum(ere_connectedness_scores_one_step)
            two_step_sum = sum(ere_connectedness_scores_two_step)

            one_step_max = max(ere_connectedness_scores_one_step)

            shared_ere_type_ids_list.append([ere_type, ere_ids, two_step_sum, one_step_sum, two_step_min, one_step_min, one_step_max])

    shared_ere_type_ids_list = sorted(shared_ere_type_ids_list, key=itemgetter(2, 3, 4, 5), reverse=True)

    if not shared_ere_type_ids_list:
        return None

    used_ids = set()

    top_shared_ere_type_ids_list = []
    graph_idx = random.choice(range(len(graphs)))

    first_entry = shared_ere_type_ids_list[0]
    top_shared_ere_type_ids_list.append(first_entry)
    used_ids.update(first_entry[1])
    target_ere_id = first_entry[1][graph_idx]

    for _ in range(num_shared_eres - 1):
        shared_ere_type_ids_list = filter_out_used(shared_ere_type_ids_list, used_ids)

        if not shared_ere_type_ids_list:
            return None

        while not reachable(graphs[graph_idx], target_ere_id, shared_ere_type_ids_list[0][1][graph_idx]):
            shared_ere_type_ids_list = shared_ere_type_ids_list[1:]
            if not shared_ere_type_ids_list:
                return None

        top_shared_ere_type_ids_list.append(shared_ere_type_ids_list[0])
        used_ids.update(shared_ere_type_ids_list[0][1])

    top_shared_ere_type_ids_list = sorted(top_shared_ere_type_ids_list, key=itemgetter(3), reverse=True)

    return list(map(itemgetter(0, 1), top_shared_ere_type_ids_list)), graph_idx

def filter_out_used(shared_ere_type_ids_list, used_ids):
    rem_list = [iter for iter, item in enumerate([temp[1] for temp in shared_ere_type_ids_list]) if set.intersection(set(item), used_ids)]
    return [item for iter, item in enumerate(shared_ere_type_ids_list) if iter not in rem_list]


def reachable(graph, target_ere_id, new_ere_id):
    seen_eres = set()
    curr_eres = [target_ere_id]

    while new_ere_id not in seen_eres:
        seen_eres.update(curr_eres)
        next_eres = set([item for neighs in [graph.eres[ere_id].neighbor_ere_ids for ere_id in curr_eres] for item in neighs]) - seen_eres
        if not next_eres and new_ere_id not in seen_eres:
            return False

        curr_eres = list(next_eres)

    return True


def replace_ere(graph, source_id, target_ere):
    """Replace the `source_id` node in `graph` with the `target_node`."""
    neighbor_ere_ids = graph.eres[source_id].neighbor_ere_ids.copy()
    stmt_ids = graph.eres[source_id].stmt_ids.copy()
    del graph.eres[source_id]
    graph.eres[target_ere.id] = deepcopy(target_ere)
    graph.eres[target_ere.id].neighbor_ere_ids = neighbor_ere_ids
    graph.eres[target_ere.id].stmt_ids = stmt_ids

    for stmt_id in stmt_ids:
        if graph.stmts[stmt_id].tail_id == source_id:
            graph.stmts[stmt_id].tail_id = target_ere.id
            graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids.discard(source_id)
            graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids.add(target_ere.id)
        if graph.stmts[stmt_id].head_id == source_id:
            graph.stmts[stmt_id].head_id = target_ere.id
            if graph.stmts[stmt_id].tail_id is not None:
                graph.eres[graph.stmts[stmt_id].tail_id].neighbor_ere_ids.discard(source_id)
                graph.eres[graph.stmts[stmt_id].tail_id].neighbor_ere_ids.add(target_ere.id)


# This function mixes a set of single-doc graphs at num_shared_eres points.
def create_mix(first_graph_list, second_graph_list, event_names, entity_names, self_pair_entity_names, event_type_maps, entity_type_maps, allow_rel_query, one_step_connectedness_map, two_step_connectedness_map,
                                        num_sources, num_query_hops, query_decay, min_connectedness_one_step, min_connectedness_two_step, used_pairs, pos, tried_names, tried_ere_ids):
    decay_rate = query_decay

    if pos:
        try:
            ent_ere_id = random.sample(list(self_pair_entity_names), 1)[0]

            graph = deepcopy(first_graph_list[ent_ere_id.split('_h')[0]])
            orig_graph = deepcopy(graph)

            seen_eres = set()
            curr_eres = {ent_ere_id}

            while len(curr_eres) > 0:
                seen_eres.update(curr_eres)

                curr_eres = set.union(*[graph.eres[ere_id].neighbor_ere_ids for ere_id in curr_eres]) - seen_eres

            reachable_stmts = set.union(*[graph.eres[ere_id].stmt_ids for ere_id in seen_eres])

            for ere_id in set(graph.eres.keys()) - seen_eres:
                del graph.eres[ere_id]

            for stmt_id in set(graph.stmts.keys()) - reachable_stmts:
                del graph.stmts[stmt_id]

            event_stmts = set([stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Event') and (len(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)])
            rel_stmts = set([stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation') and (len(graph.eres[list(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids - {ent_ere_id})[0]].neighbor_ere_ids) > 1)])

            poss_pairs = []

            for stmt_1, stmt_2 in [item for item in itertools.combinations(list(set.union(event_stmts, rel_stmts)), 2) if 'Event' in {graph.eres[graph.stmts[sub_item].head_id].category for sub_item in item}]:
                if {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Event'}:
                    inter_eres = set.intersection(graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids, graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids)
                    neighs_1 = graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids - inter_eres
                    neighs_2 = graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids - inter_eres

                    if len(inter_eres - {ent_ere_id}) >= max(0, (1 - len(neighs_1))) + max(0, (1 - len(neighs_2))):
                        if check_distinct_ere_pair(graph, graph.stmts[stmt_1].head_id, graph.stmts[stmt_2].head_id):
                            poss_pairs.append((stmt_1, stmt_2, 2))
                elif allow_rel_query and {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Event', 'Relation'}:
                    if graph.eres[graph.stmts[stmt_2].head_id].category == 'Event':
                        temp = stmt_2
                        stmt_2 = stmt_1
                        stmt_1 = temp

                    rel_ere = graph.eres[graph.stmts[stmt_2].head_id]
                    event_neighs = graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids
                    other_rel_neigh = list(rel_ere.neighbor_ere_ids - {ent_ere_id})[0]
                    rel_ev_neighs = {item for item in graph.eres[other_rel_neigh].neighbor_ere_ids if graph.eres[item].category == 'Event'}
                    rel_rel_neighs = set.union(*[(graph.eres[item].neighbor_ere_ids - {other_rel_neigh}) for item in graph.eres[other_rel_neigh].neighbor_ere_ids if graph.eres[item].category == 'Relation'])

                    if graph.stmts[stmt_1].head_id in rel_ev_neighs:
                        continue

                    rel_neighs = set.union(rel_ev_neighs, rel_rel_neighs)

                    inter_eres = set.intersection(event_neighs, rel_neighs)

                    neighs_1 = event_neighs - inter_eres
                    neighs_2 = rel_neighs - inter_eres

                    if len(inter_eres - {ent_ere_id}) >= max(0, (1 - len(neighs_1))) + max(0, (1 - len(neighs_2))):
                        if check_distinct_ere_pair(graph, graph.stmts[stmt_1].head_id, graph.stmts[stmt_2].head_id):
                            poss_pairs.append((stmt_1, stmt_2, 0))

            stmt_1, stmt_2, target_ind = random.sample(poss_pairs, 1)[0]

            if target_ind == 2:
                target_event_stmt = random.sample([stmt_1, stmt_2], 1)[0]
                inter_eres = set.intersection(graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids, graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids)
                target_event_neighs = graph.eres[graph.stmts[target_event_stmt].head_id].neighbor_ere_ids - inter_eres
                other_event_stmt = list(set([stmt_1, stmt_2]) - {target_event_stmt})[0]
                other_event_neighs = graph.eres[graph.stmts[other_event_stmt].head_id].neighbor_ere_ids - inter_eres
            elif target_ind == 0:
                target_event_stmt = stmt_1
                rel_ere = graph.eres[graph.stmts[stmt_2].head_id]
                event_neighs = graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids
                other_rel_neigh = list(rel_ere.neighbor_ere_ids - {ent_ere_id})[0]
                rel_ev_neighs = {item for item in graph.eres[other_rel_neigh].neighbor_ere_ids if graph.eres[item].category == 'Event'}
                rel_rel_neighs = set.union(*[(graph.eres[item].neighbor_ere_ids - {other_rel_neigh}) for item in graph.eres[other_rel_neigh].neighbor_ere_ids if graph.eres[item].category == 'Relation'])
                rel_neighs = set.union(rel_ev_neighs, rel_rel_neighs)
                inter_eres = set.intersection(event_neighs, rel_neighs)
                target_event_neighs = event_neighs - inter_eres
                other_event_neighs = rel_neighs - inter_eres

            target_event_ere_id = graph.stmts[target_event_stmt].head_id

            other_event_stmt = list({stmt_1, stmt_2} - {target_event_stmt})[0]

            if graph.eres[graph.stmts[other_event_stmt].head_id].category == 'Event':
                other_event_ere_id = graph.stmts[other_event_stmt].head_id
            elif graph.eres[graph.stmts[other_event_stmt].head_id].category == 'Relation':
                other_event_ere_id = list(graph.eres[graph.stmts[other_event_stmt].head_id].neighbor_ere_ids - {ent_ere_id})[0]

            ere_pool = deepcopy(inter_eres - {ent_ere_id})

            other_eres = set(random.sample(list(ere_pool), max(0, (1 - len(other_event_neighs)))))
            ere_pool -= other_eres
            target_eres = set(random.sample(list(ere_pool), max(0, (1 - len(target_event_neighs)))))
            ere_pool -= target_eres

            other = True

            while len(ere_pool) > 0:
                if other:
                    other_eres.update(set(random.sample(list(ere_pool), 1)))
                else:
                    target_eres.update(set(random.sample(list(ere_pool), 1)))

                ere_pool -= set.union(other_eres, target_eres)

                other = not other

            rem_event_stmts = set([stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Event')]) - set([target_event_stmt, other_event_stmt])
            rem_rel_stmts = set([stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation')]) - {other_event_stmt}

            focus_span = get_stmt_spans(graph)

            add_other_event_stmts = set()
            add_other_rel_stmts = set()

            if graph.eres[graph.stmts[other_event_stmt].head_id].category == 'Event':
                add_other_event_stmts.add(other_event_stmt)
            else:
                add_other_rel_stmts.update({item for item in graph.eres[graph.stmts[other_event_stmt].head_id].stmt_ids if graph.stmts[item].tail_id})

            add_target_event_stmts = {target_event_stmt}
            add_target_rel_stmts = set()

            other = True

            while len(rem_event_stmts) > 0:
                if other:
                    poss_options = {item for item in rem_event_stmts if not set.intersection(focus_span[item], set.union(*[focus_span[stmt_id] for stmt_id in add_target_event_stmts]))}

                    if poss_options:
                        add_other_event_stmts.update(random.sample(list(poss_options), 1))
                    else:
                        add_target_event_stmts.update({item for item in rem_event_stmts if not set.intersection(focus_span[item], set.union(*[focus_span[stmt_id] for stmt_id in set.union(add_other_event_stmts, add_other_rel_stmts)]))})
                        break
                else:
                    poss_options = {item for item in rem_event_stmts if not set.intersection(focus_span[item], set.union(*[focus_span[stmt_id] for stmt_id in set.union(add_other_event_stmts, add_other_rel_stmts)]))}

                    if poss_options:
                        add_target_event_stmts.update(random.sample(list(poss_options), 1))
                    else:
                        add_other_event_stmts.update({item for item in rem_event_stmts if not set.intersection(focus_span[item], set.union(*[focus_span[stmt_id] for stmt_id in add_target_event_stmts]))})
                        break

                rem_event_stmts -= set.union(add_other_event_stmts, add_target_event_stmts)

                other = not other

            if len(add_other_event_stmts) > len(add_target_event_stmts):
                other = False
            else:
                other = True

            add_other_event_ere_ids = set()
            add_other_rel_ere_ids = set()

            if add_other_event_stmts:
                add_other_event_ere_ids = {graph.stmts[item].head_id for item in add_other_event_stmts}

            if add_target_event_stmts:
                add_target_event_ere_ids = {graph.stmts[item].head_id for item in add_target_event_stmts}

            rel_neigh_ids_to_add = set()

            if add_other_rel_stmts:
                rel_neigh_ids_to_add = set.union(*[graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ent_ere_id} for item in add_other_rel_stmts])

            avoid_eres_other = set.union(other_eres, {other_event_ere_id}, other_event_neighs, add_other_event_ere_ids, rel_neigh_ids_to_add)
            avoid_eres_target = set.union(target_eres, {target_event_ere_id}, target_event_neighs, add_target_event_ere_ids)

            last_len = -1

            while (len(rem_rel_stmts) > 0) and len(rem_rel_stmts) != last_len:
                last_len = len(rem_rel_stmts)

                if other:
                    poss_options = {item for item in rem_rel_stmts if not set.intersection(focus_span[item], set.union(*[focus_span[stmt_id] for stmt_id in set.union(add_target_event_stmts, add_target_rel_stmts)])) and
                                    not set.intersection(graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ent_ere_id}, avoid_eres_target)}

                    if poss_options:
                        add_stmt = random.sample(list(poss_options), 1)[0]
                        add_other_rel_stmts.update({item for item in graph.eres[graph.stmts[add_stmt].head_id].stmt_ids if graph.stmts[item].tail_id})
                    else:
                        add_stmts = {item for item in rem_rel_stmts if not set.intersection(focus_span[item], set.union(*[focus_span[stmt_id] for stmt_id in set.union(add_other_event_stmts, add_other_rel_stmts)])) and
                                     not set.intersection(graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ent_ere_id}, avoid_eres_other)}

                        if add_stmts:
                            add_target_rel_stmts.update(set.union(*[{item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id} for stmt_id in add_stmts]))

                            rem_rel_stmts -= set.union(add_other_rel_stmts, add_target_rel_stmts)
                        break
                else:
                    poss_options = {item for item in rem_rel_stmts if not set.intersection(focus_span[item], set.union(*[focus_span[stmt_id] for stmt_id in set.union(add_other_event_stmts, add_other_rel_stmts)])) and
                                    not set.intersection(graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ent_ere_id}, avoid_eres_other)}

                    if poss_options:
                        add_stmt = random.sample(list(poss_options), 1)[0]
                        add_target_rel_stmts.update({item for item in graph.eres[graph.stmts[add_stmt].head_id].stmt_ids if graph.stmts[item].tail_id})
                    else:
                        add_stmts = {item for item in rem_rel_stmts if not set.intersection(focus_span[item], set.union(*[focus_span[stmt_id] for stmt_id in set.union(add_target_event_stmts, add_target_rel_stmts)])) and
                                     not set.intersection(graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ent_ere_id}, avoid_eres_target)}

                        if add_stmts:
                            add_other_rel_stmts.update(set.union(*[{item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id} for stmt_id in add_stmts]))

                            rem_rel_stmts -= set.union(add_other_rel_stmts, add_target_rel_stmts)
                        break

                if add_other_rel_stmts:
                    avoid_eres_other.update(set.union(*[graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ent_ere_id} for item in add_other_rel_stmts]))

                if add_target_rel_stmts:
                    avoid_eres_target.update(set.union(*[graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ent_ere_id} for item in add_target_rel_stmts]))

                rem_rel_stmts -= set.union(add_other_rel_stmts, add_target_rel_stmts)

                other = not other

            for stmt_id in rem_rel_stmts:
                rel_stmt_ids = {item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id}

                for neigh_id in graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids:
                    graph.eres[neigh_id].stmt_ids -= rel_stmt_ids

                    if {item for item in graph.eres[neigh_id].stmt_ids if graph.stmts[item].tail_id}:
                            graph.eres[neigh_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[neigh_id].stmt_ids if graph.stmts[item].tail_id]) - {neigh_id}
                    else:
                        del graph.eres[neigh_id]

                del graph.eres[graph.stmts[stmt_id].head_id]

                for other_stmt_id in rel_stmt_ids:
                    del graph.stmts[other_stmt_id]

            assert not set.intersection(set.union(*[focus_span[item] for item in set.union(add_target_event_stmts, add_target_rel_stmts)]), set.union(*[focus_span[item] for item in set.union(add_other_event_stmts, add_other_rel_stmts)]))

            if add_other_rel_stmts:
                add_other_rel_ere_ids = set.union(*[graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ent_ere_id} for item in add_other_rel_stmts])

            res_stmts = set()

            avoid_eres = set.union(other_eres, {other_event_ere_id}, other_event_neighs, add_other_event_ere_ids, add_other_rel_ere_ids)
            seen_eres = deepcopy(avoid_eres)

            curr_eres = {ent_ere_id}

            curr_num_hops = 0

            while curr_num_hops < 3 and len(curr_eres) > 0:
                seen_eres.update(curr_eres)
                stmt_ids = set.union(*[set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id]) for ere_id in curr_eres])

                for stmt_id in stmt_ids:
                    if not set.intersection(focus_span[stmt_id], set.union(*[focus_span[item] for item in set.union(add_other_event_stmts, add_other_rel_stmts)])):
                        if not set.intersection(avoid_eres, {graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id}):
                            if graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
                                if not set.intersection(set.union(*[{graph.stmts[neigh_stmt_id].head_id, graph.stmts[neigh_stmt_id].tail_id} for neigh_stmt_id in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[neigh_stmt_id].tail_id]), avoid_eres):
                                    res_stmts.add(stmt_id)
                            else:
                                res_stmts.add(stmt_id)

                curr_eres = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in res_stmts if graph.stmts[stmt_id].tail_id]) - seen_eres

                curr_num_hops += 1

            res_eres = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in res_stmts if graph.stmts[stmt_id].tail_id]) - {ent_ere_id}

            for res_ere in res_eres:
                if graph.eres[res_ere].category == 'Relation':
                    res_stmts.update([item for item in graph.eres[res_ere].stmt_ids if graph.stmts[item].tail_id])

            res_eres = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in res_stmts if graph.stmts[stmt_id].tail_id]) - {ent_ere_id}

            assert not set.intersection(set.union(*[focus_span[item] for item in res_stmts]), set.union(*[focus_span[item] for item in set.union(add_other_event_stmts, add_other_rel_stmts)]))
            assert len(set.intersection(set.union(add_target_event_stmts, add_target_rel_stmts), res_stmts)) == len(set.union(add_target_event_stmts, add_target_rel_stmts))

            old_res_stmts = deepcopy(res_stmts)

            all_stmt_diff = set()

            for ere_id in res_eres:
                all_stmt_diff.update([item for item in (graph.eres[ere_id].stmt_ids - res_stmts) if graph.stmts[item].tail_id])

            poss_new_res_eres = deepcopy(res_eres)
            poss_new_res_stmts = deepcopy(res_stmts)

            for stmt_diff_id in all_stmt_diff:
                if graph.stmts[stmt_diff_id].head_id in res_eres:
                    other_ere_id = graph.stmts[stmt_diff_id].tail_id
                elif graph.stmts[stmt_diff_id].tail_id in res_eres:
                    other_ere_id = graph.stmts[stmt_diff_id].head_id

                seen_eres = deepcopy(res_eres)
                curr_eres = {other_ere_id}

                while len(curr_eres) > 0:
                    seen_eres.update(curr_eres)

                    neigh_eres = set()

                    for ere_id in curr_eres:
                        stmt_neighs_to_add = {stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id}

                        if stmt_neighs_to_add:
                            neigh_eres.update(set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} - {ere_id} for stmt_id in stmt_neighs_to_add]))

                    curr_eres = neigh_eres - seen_eres

                if ent_ere_id not in seen_eres:
                    poss_new_res_eres.update(seen_eres)

            poss_new_res_stmts.update({stmt_id for stmt_id in graph.stmts.keys() if graph.stmts[stmt_id].tail_id and len(set.intersection({graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id}, poss_new_res_eres)) == 2
                              and not set.intersection(focus_span[stmt_id], set.union(*[focus_span[item] for item in set.union(add_other_event_stmts, add_other_rel_stmts)]))})

            seen_eres = set()
            curr_eres = {ent_ere_id}

            while len(curr_eres) > 0:
                seen_eres.update(curr_eres)

                neigh_eres = set()

                for ere_id in curr_eres:
                    stmt_neighs_to_add = {stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and stmt_id in poss_new_res_stmts}

                    if stmt_neighs_to_add:
                        neigh_eres.update(set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} - {ere_id} for stmt_id in stmt_neighs_to_add]))

                curr_eres = neigh_eres - seen_eres

            res_eres = seen_eres

            res_stmts.update({stmt_id for stmt_id in graph.stmts.keys() if graph.stmts[stmt_id].tail_id and len(set.intersection({graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id}, res_eres)) == 2 and stmt_id in poss_new_res_stmts})
            res_stmts.update({stmt_id for stmt_id in graph.stmts.keys() if not graph.stmts[stmt_id].tail_id and graph.stmts[stmt_id].head_id in res_eres})

            for stmt_id in poss_new_res_stmts - res_stmts:
                stmt = graph.stmts[stmt_id]

                graph.eres[stmt.head_id].stmt_ids.discard(stmt_id)
                if {item for item in graph.eres[stmt.head_id].stmt_ids if graph.stmts[item].tail_id}:
                    graph.eres[stmt.head_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[stmt.head_id].stmt_ids if graph.stmts[item].tail_id]) - {stmt.head_id}

                graph.eres[stmt.tail_id].stmt_ids.discard(stmt_id)
                if {item for item in graph.eres[stmt.tail_id].stmt_ids if graph.stmts[item].tail_id}:
                    graph.eres[stmt.tail_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[stmt.tail_id].stmt_ids if graph.stmts[item].tail_id]) - {stmt.tail_id}

                del graph.stmts[stmt_id]

            for stmt_id in {res_stmt for res_stmt in res_stmts if graph.stmts[res_stmt].tail_id}:
                stmt = graph.stmts[stmt_id]

                assert set.intersection({item for item in set.union(graph.eres[stmt.head_id].stmt_ids, graph.eres[stmt.tail_id].stmt_ids) if graph.stmts[item].tail_id} - {stmt_id}, res_stmts)

            assert not set.intersection(set.union(*[focus_span[item] for item in {stmt_id for stmt_id in res_stmts if graph.stmts[stmt_id].tail_id}]), set.union(*[focus_span[item] for item in set.union(add_other_event_stmts, add_other_rel_stmts)]))

            for stmt_id in res_stmts:
                stmt = graph.stmts[stmt_id]

                if stmt.tail_id:
                    assert len(set.intersection({stmt.head_id, stmt.tail_id}, res_eres)) == 2

            rej_ere_ids = get_rej_ere_ids(graph, {item for item in res_stmts if graph.stmts[item].tail_id})

            poss_res_stmts = set()

            res_eres.discard(ent_ere_id)

            all_stmt_diff = set()

            for ere_id in res_eres:
                stmt_diff = {stmt_id for stmt_id in (graph.eres[ere_id].stmt_ids - res_stmts) if graph.stmts[stmt_id].tail_id}
                all_stmt_diff.update(stmt_diff)

                for stmt_id in stmt_diff:
                    other_neigh = list({graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} - {ere_id})[0]

                    if other_neigh in rej_ere_ids:
                        poss_res_stmts.update({item for item in graph.eres[other_neigh].stmt_ids if graph.stmts[item].tail_id})
                    elif graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
                        poss_res_stmts.update({item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id})
                    elif graph.eres[graph.stmts[stmt_id].head_id].category == 'Event':
                        poss_res_stmts.add(stmt_id)

            poss_res_stmts_to_add = check_poss_stmt_span(graph, poss_res_stmts, res_stmts, rej_ere_ids, add_other_event_stmts, add_other_rel_stmts)

            poss_res_stmts_to_rem = set()

            for rej_ere_id in rej_ere_ids:
                if not set.intersection({item for item in graph.eres[rej_ere_id].stmt_ids if graph.stmts[item].tail_id and set.intersection({graph.stmts[item].head_id, graph.stmts[item].tail_id}, res_eres)}, poss_res_stmts_to_add):
                    poss_res_stmts_to_rem.update(graph.eres[rej_ere_id].stmt_ids)

                    stmts_to_dis = graph.eres[rej_ere_id].stmt_ids

                    for stmt_id in graph.eres[rej_ere_id].stmt_ids:
                        del graph.stmts[stmt_id]

                    del graph.eres[rej_ere_id]

                    for ere_id in graph.eres.keys():
                        graph.eres[ere_id].stmt_ids -= stmts_to_dis
                        graph.eres[ere_id].neighbor_ere_ids -= {rej_ere_id}

            poss_res_stmts_to_add -= poss_res_stmts_to_rem
            rej_ere_ids = set.intersection(rej_ere_ids, set(graph.eres.keys()))

            assert check_spans_distinct(graph, (poss_res_stmts - poss_res_stmts_to_rem), {item for item in res_stmts if graph.stmts[item].tail_id})

            if len({stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and stmt_id not in res_stmts}) == 0:
                return False

            rej_eres_to_dup_cand = defaultdict(set)
            eres_to_dup_cand = defaultdict(set)
            eres_to_dup_query = defaultdict(set)

            for ere_id in res_eres:
                stmt_diff = {stmt_id for stmt_id in (graph.eres[ere_id].stmt_ids - res_stmts) if graph.stmts[stmt_id].tail_id}

                for stmt_id in stmt_diff:
                    if graph.stmts[stmt_id].head_id in rej_ere_ids:
                        stmt_ids = {item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id}

                        rej_eres_to_dup_cand[graph.stmts[stmt_id].head_id].update(set.intersection(stmt_ids, poss_res_stmts_to_add))
                    elif graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
                        other_stmt_id = list({item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id} - {stmt_id})[0]

                        if set.intersection(graph.eres[graph.stmts[stmt_id].head_id].stmt_ids, poss_res_stmts_to_add):
                            query_side_ere_id = graph.stmts[other_stmt_id].tail_id
                            eres_to_dup_cand[query_side_ere_id].add(stmt_id)
                        else:
                            eres_to_dup_query[ere_id].add(other_stmt_id)
                    elif graph.eres[graph.stmts[stmt_id].head_id].category == 'Event':
                        if stmt_id in poss_res_stmts_to_add:
                            query_side_ere_id = list({graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} - {ere_id})[0]
                            eres_to_dup_cand[query_side_ere_id].add(stmt_id)
                        else:
                            eres_to_dup_query[ere_id].add(stmt_id)

            for ere_id in rej_eres_to_dup_cand.keys():
                stmt_ids = rej_eres_to_dup_cand[ere_id]

                for stmt_id in [item for item in stmt_ids if graph.stmts[item].tail_id in res_eres]:
                    rej_ere_neigh = graph.stmts[stmt_id].head_id

                    if rej_ere_neigh + '_dup' not in graph.eres.keys():
                        graph.eres[rej_ere_neigh + '_dup'] = deepcopy(graph.eres[rej_ere_neigh])

                        for item in graph.eres[rej_ere_neigh].stmt_ids:
                            if not graph.stmts[item].tail_id:
                                graph.stmts[item + '_dup'] = deepcopy(graph.stmts[item])

                                graph.stmts[item + '_dup'].head_id = rej_ere_neigh + '_dup'

                        graph.eres[rej_ere_neigh + '_dup'].stmt_ids = {item + '_dup' for item in graph.eres[rej_ere_neigh].stmt_ids if not graph.stmts[item].tail_id}

                    graph.eres[rej_ere_neigh].stmt_ids.discard(stmt_id)

                    if {item for item in graph.eres[rej_ere_neigh].stmt_ids if graph.stmts[item].tail_id}:
                        graph.eres[rej_ere_neigh].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[rej_ere_neigh].stmt_ids if graph.stmts[item].tail_id]) - {rej_ere_neigh}

                    graph.eres[rej_ere_neigh + '_dup'].stmt_ids.add(stmt_id)
                    graph.eres[rej_ere_neigh + '_dup'].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[rej_ere_neigh + '_dup'].stmt_ids if graph.stmts[item].tail_id]) - {rej_ere_neigh, rej_ere_neigh + '_dup'}
                    graph.stmts[stmt_id].head_id = rej_ere_neigh + '_dup'

                    graph.eres[graph.stmts[stmt_id].tail_id].neighbor_ere_ids.discard(rej_ere_neigh)
                    graph.eres[graph.stmts[stmt_id].tail_id].neighbor_ere_ids.add(rej_ere_neigh + '_dup')

                    res_stmts.update(graph.eres[rej_ere_neigh + '_dup'].stmt_ids)
                    res_eres.add(rej_ere_neigh + '_dup')

                for stmt_id in [item for item in stmt_ids if graph.stmts[item].tail_id not in res_eres]:
                    tail_id = graph.stmts[stmt_id].tail_id

                    if tail_id + '_dup' not in graph.eres.keys():
                        graph.eres[tail_id + '_dup'] = deepcopy(graph.eres[tail_id])

                        for item in graph.eres[tail_id].stmt_ids:
                            if not graph.stmts[item].tail_id:
                                graph.stmts[item + '_dup'] = deepcopy(graph.stmts[item])

                                graph.stmts[item + '_dup'].head_id = tail_id + '_dup'

                        graph.eres[tail_id  + '_dup'].stmt_ids = {item + '_dup' for item in graph.eres[tail_id].stmt_ids if not graph.stmts[item].tail_id}

                    graph.eres[tail_id].stmt_ids.discard(stmt_id)

                    if {item for item in graph.eres[tail_id].stmt_ids if graph.stmts[item].tail_id}:
                        graph.eres[tail_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[tail_id].stmt_ids if graph.stmts[item].tail_id]) - {tail_id}

                    graph.stmts[stmt_id].head_id = graph.stmts[stmt_id].head_id + '_dup'

                    graph.eres[tail_id + '_dup'].stmt_ids.add(stmt_id)
                    graph.eres[tail_id + '_dup'].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[tail_id + '_dup'].stmt_ids if graph.stmts[item].tail_id]) - {tail_id, tail_id + '_dup'}
                    graph.stmts[stmt_id].tail_id = tail_id + '_dup'

                    graph.eres[graph.stmts[stmt_id].head_id].stmt_ids.add(stmt_id)
                    graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids.add(tail_id + '_dup')

                    orig_head_id = graph.stmts[stmt_id].head_id.split('_dup')[0]
                    graph.eres[orig_head_id].stmt_ids.discard(stmt_id)

                    if {item for item in graph.eres[orig_head_id].stmt_ids if graph.stmts[item].tail_id}:
                        graph.eres[orig_head_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[orig_head_id].stmt_ids if graph.stmts[item].tail_id]) - {orig_head_id}

                    res_stmts.update(graph.eres[graph.stmts[stmt_id].tail_id].stmt_ids)
                    res_eres.add(graph.stmts[stmt_id].tail_id)

            for ere_id in eres_to_dup_cand.keys():
                stmt_ids = eres_to_dup_cand[ere_id]

                for stmt_id in stmt_ids:
                    if graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
                        other_stmt_id = list({item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id} - {stmt_id})[0]
                        query_side_ere_id = graph.stmts[other_stmt_id].tail_id

                        if query_side_ere_id + '_dup' not in graph.eres.keys():
                            graph.eres[query_side_ere_id + '_dup'] = deepcopy(graph.eres[query_side_ere_id])

                            for item in graph.eres[query_side_ere_id].stmt_ids:
                                if not graph.stmts[item].tail_id:
                                    graph.stmts[item + '_dup'] = deepcopy(graph.stmts[item])

                                    graph.stmts[item + '_dup'].head_id = query_side_ere_id + '_dup'

                            graph.eres[query_side_ere_id + '_dup'].stmt_ids = {item + '_dup' for item in graph.eres[query_side_ere_id].stmt_ids if not graph.stmts[item].tail_id}

                        graph.eres[query_side_ere_id].stmt_ids.discard(other_stmt_id)

                        if {item for item in graph.eres[query_side_ere_id].stmt_ids if graph.stmts[item].tail_id}:
                            graph.eres[query_side_ere_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[query_side_ere_id].stmt_ids if graph.stmts[item].tail_id]) - {query_side_ere_id}

                        graph.eres[query_side_ere_id + '_dup'].stmt_ids.add(other_stmt_id)
                        graph.eres[query_side_ere_id + '_dup'].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[query_side_ere_id + '_dup'].stmt_ids if graph.stmts[item].tail_id]) - {query_side_ere_id, query_side_ere_id + '_dup'}
                        graph.stmts[other_stmt_id].tail_id = query_side_ere_id + '_dup'

                        graph.eres[graph.stmts[other_stmt_id].head_id].neighbor_ere_ids.discard(query_side_ere_id)
                        graph.eres[graph.stmts[other_stmt_id].head_id].neighbor_ere_ids.add(query_side_ere_id + '_dup')

                        res_stmts.update(graph.eres[graph.stmts[stmt_id].head_id].stmt_ids)
                        res_stmts.update(graph.eres[query_side_ere_id + '_dup'].stmt_ids)
                        res_eres.update([query_side_ere_id + '_dup', graph.stmts[stmt_id].head_id])
                    elif graph.eres[graph.stmts[stmt_id].head_id].category == 'Event':
                        query_side_ere_id = ere_id

                        if query_side_ere_id + '_dup' not in graph.eres.keys():
                            graph.eres[query_side_ere_id + '_dup'] = deepcopy(graph.eres[query_side_ere_id])

                            for item in graph.eres[query_side_ere_id].stmt_ids:
                                if not graph.stmts[item].tail_id:
                                    graph.stmts[item + '_dup'] = deepcopy(graph.stmts[item])

                                    graph.stmts[item + '_dup'].head_id = query_side_ere_id + '_dup'

                            graph.eres[query_side_ere_id + '_dup'].stmt_ids = {item + '_dup' for item in graph.eres[query_side_ere_id].stmt_ids if not graph.stmts[item].tail_id}

                        graph.eres[query_side_ere_id].stmt_ids.discard(stmt_id)

                        if {item for item in graph.eres[query_side_ere_id].stmt_ids if graph.stmts[item].tail_id}:
                            graph.eres[query_side_ere_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[query_side_ere_id].stmt_ids if graph.stmts[item].tail_id]) - {query_side_ere_id}

                        graph.eres[query_side_ere_id + '_dup'].stmt_ids.add(stmt_id)
                        graph.eres[query_side_ere_id + '_dup'].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[query_side_ere_id + '_dup'].stmt_ids if graph.stmts[item].tail_id]) - {query_side_ere_id, query_side_ere_id + '_dup'}

                        if graph.stmts[stmt_id].head_id == query_side_ere_id:
                            graph.stmts[stmt_id].head_id = query_side_ere_id + '_dup'
                            cand_side_ere_id = graph.stmts[stmt_id].tail_id
                        elif graph.stmts[stmt_id].tail_id == query_side_ere_id:
                            graph.stmts[stmt_id].tail_id = query_side_ere_id + '_dup'
                            cand_side_ere_id = graph.stmts[stmt_id].head_id

                        graph.eres[cand_side_ere_id].neighbor_ere_ids.discard(query_side_ere_id)
                        graph.eres[cand_side_ere_id].neighbor_ere_ids.add(query_side_ere_id + '_dup')

                        res_stmts.update(graph.eres[query_side_ere_id + '_dup'].stmt_ids)
                        res_eres.add(query_side_ere_id + '_dup')

            for ere_id in eres_to_dup_query.keys():
                stmt_ids = eres_to_dup_query[ere_id]

                for stmt_id in stmt_ids:
                    if graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
                        other_stmt_id = list({item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id} - {stmt_id})[0]
                        cand_side_ere_id = graph.stmts[other_stmt_id].tail_id

                        if cand_side_ere_id + '_dup' not in graph.eres.keys():
                            graph.eres[cand_side_ere_id + '_dup'] = deepcopy(graph.eres[cand_side_ere_id])

                            for item in graph.eres[cand_side_ere_id].stmt_ids:
                                if not graph.stmts[item].tail_id:
                                    graph.stmts[item + '_dup'] = deepcopy(graph.stmts[item])

                                    graph.stmts[item + '_dup'].head_id = cand_side_ere_id + '_dup'

                            graph.eres[cand_side_ere_id + '_dup'].stmt_ids = {item + '_dup' for item in graph.eres[cand_side_ere_id].stmt_ids if not graph.stmts[item].tail_id}

                        graph.eres[cand_side_ere_id].stmt_ids.discard(other_stmt_id)

                        if {item for item in graph.eres[cand_side_ere_id].stmt_ids if graph.stmts[item].tail_id}:
                            graph.eres[cand_side_ere_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[cand_side_ere_id].stmt_ids if graph.stmts[item].tail_id]) - {cand_side_ere_id}

                        graph.eres[cand_side_ere_id + '_dup'].stmt_ids.add(other_stmt_id)
                        graph.eres[cand_side_ere_id + '_dup'].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[cand_side_ere_id + '_dup'].stmt_ids if graph.stmts[item].tail_id]) - {cand_side_ere_id, cand_side_ere_id + '_dup'}
                        graph.stmts[other_stmt_id].tail_id = cand_side_ere_id + '_dup'

                        graph.eres[graph.stmts[other_stmt_id].head_id].neighbor_ere_ids.discard(cand_side_ere_id)
                        graph.eres[graph.stmts[other_stmt_id].head_id].neighbor_ere_ids.add(cand_side_ere_id + '_dup')
                    elif graph.eres[graph.stmts[stmt_id].head_id].category == 'Event':
                        cand_side_ere_id = ere_id

                        if cand_side_ere_id + '_dup' not in graph.eres.keys():
                            graph.eres[cand_side_ere_id + '_dup'] = deepcopy(graph.eres[cand_side_ere_id])

                            for item in graph.eres[cand_side_ere_id].stmt_ids:
                                if not graph.stmts[item].tail_id:
                                    graph.stmts[item + '_dup'] = deepcopy(graph.stmts[item])

                                    graph.stmts[item + '_dup'].head_id = cand_side_ere_id + '_dup'

                            graph.eres[cand_side_ere_id + '_dup'].stmt_ids = {item + '_dup' for item in graph.eres[cand_side_ere_id].stmt_ids if not graph.stmts[item].tail_id}

                        graph.eres[cand_side_ere_id].stmt_ids.discard(stmt_id)

                        if {item for item in graph.eres[cand_side_ere_id].stmt_ids if graph.stmts[item].tail_id}:
                            graph.eres[cand_side_ere_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[cand_side_ere_id].stmt_ids if graph.stmts[item].tail_id]) - {cand_side_ere_id}

                        graph.eres[cand_side_ere_id + '_dup'].stmt_ids.add(stmt_id)
                        graph.eres[cand_side_ere_id + '_dup'].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[cand_side_ere_id + '_dup'].stmt_ids if graph.stmts[item].tail_id]) - {cand_side_ere_id, cand_side_ere_id + '_dup'}

                        if graph.stmts[stmt_id].head_id == cand_side_ere_id:
                            graph.stmts[stmt_id].head_id = cand_side_ere_id + '_dup'
                            query_side_ere_id = graph.stmts[stmt_id].tail_id
                        elif graph.stmts[stmt_id].tail_id == cand_side_ere_id:
                            graph.stmts[stmt_id].tail_id = cand_side_ere_id + '_dup'
                            query_side_ere_id = graph.stmts[stmt_id].head_id

                        graph.eres[query_side_ere_id].neighbor_ere_ids.discard(cand_side_ere_id)
                        graph.eres[query_side_ere_id].neighbor_ere_ids.add(cand_side_ere_id + '_dup')

            for ere_id in res_eres:
                stmt_diff = {stmt_id for stmt_id in (graph.eres[ere_id].stmt_ids - res_stmts) if graph.stmts[stmt_id].tail_id}

                graph.eres[ere_id].stmt_ids = set.union(set.intersection(res_stmts, graph.eres[ere_id].stmt_ids), {stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if not graph.stmts[stmt_id].tail_id})
                graph.eres[ere_id].neighbor_ere_ids = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id]) - {ere_id}

                for stmt_id in stmt_diff:
                    neigh_ere_id = list({graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} - {ere_id})[0]

                    graph.eres[neigh_ere_id].stmt_ids.discard(stmt_id)

                    if len([item for item in graph.eres[neigh_ere_id].stmt_ids if graph.stmts[item].tail_id]) == 0:
                        for neigh_stmt_id in graph.eres[neigh_ere_id].stmt_ids:
                            del graph.stmts[neigh_stmt_id]

                        del graph.eres[neigh_ere_id]
                    else:
                        graph.eres[neigh_ere_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[neigh_ere_id].stmt_ids if graph.stmts[item].tail_id]) - {neigh_ere_id}

                        if graph.eres[neigh_ere_id].category == 'Relation':
                            for other_stmt_id in graph.eres[neigh_ere_id].stmt_ids:
                                if graph.stmts[other_stmt_id].tail_id:
                                    other_neigh = list({graph.stmts[other_stmt_id].head_id, graph.stmts[other_stmt_id].tail_id} - {neigh_ere_id})[0]
                                    graph.eres[other_neigh].stmt_ids.discard(other_stmt_id)
                                    graph.eres[other_neigh].neighbor_ere_ids.discard(neigh_ere_id)

                                del graph.stmts[other_stmt_id]

                            del graph.eres[neigh_ere_id]

                    del graph.stmts[stmt_id]

            # seen_eres = set()
            # curr_eres = {ent_ere_id}
            #
            # while len(curr_eres) > 0:
            #     seen_eres.update(curr_eres)
            #
            #     curr_eres = set.union(*[graph.eres[ere_id].neighbor_ere_ids for ere_id in curr_eres]) - seen_eres
            #
            # reachable_stmts = set.union(*[graph.eres[ere_id].stmt_ids for ere_id in seen_eres])
            #
            # for ere_id in set(graph.eres.keys()) - seen_eres:
            #     del graph.eres[ere_id]
            #
            # for stmt_id in set(graph.stmts.keys()) - reachable_stmts:
            #     if stmt_id in graph.stmts.keys():
            #         del graph.stmts[stmt_id]

            seen_eres = set()
            curr_eres = {ent_ere_id}

            i = 0

            while len(curr_eres) > 0 and i < 5:
                seen_eres.update(curr_eres)
                curr_eres = set.union(*[graph.eres[ere_id].neighbor_ere_ids for ere_id in curr_eres]) - seen_eres

                i += 1

            reachable_stmts = set.union(*[graph.eres[ere_id].stmt_ids for ere_id in seen_eres])
            seen_eres.update(curr_eres)

            for ere_id in deepcopy(seen_eres):
                if graph.eres[ere_id].category == 'Relation':
                    reachable_stmts.update(graph.eres[ere_id].stmt_ids)
                    seen_eres.update(graph.eres[ere_id].neighbor_ere_ids)

            reachable_stmts.update(set.union(*[{item for item in graph.eres[ere_id].stmt_ids if not graph.stmts[item].tail_id} for ere_id in seen_eres]))

            for ere_id in set(graph.eres.keys()) - seen_eres:
                del graph.eres[ere_id]

            for stmt_id in set(graph.stmts.keys()) - reachable_stmts:
                del graph.stmts[stmt_id]

            for ere_id in graph.eres.keys():
                graph.eres[ere_id].stmt_ids = set.intersection(graph.eres[ere_id].stmt_ids, reachable_stmts)
                graph.eres[ere_id].neighbor_ere_ids = set.intersection(graph.eres[ere_id].neighbor_ere_ids, seen_eres)

            res_stmts = set.intersection(res_stmts, reachable_stmts)
            res_eres = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in res_stmts if graph.stmts[stmt_id].tail_id]) - {ent_ere_id}
            res_stmts.update(set.union(*[{stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if not graph.stmts[stmt_id].tail_id} for ere_id in res_eres]))

            for res_ere in res_eres:
                assert len(set.intersection({item for item in graph.eres[res_ere].stmt_ids if graph.stmts[item].tail_id}, res_stmts)) == len({item for item in graph.eres[res_ere].stmt_ids if graph.stmts[item].tail_id})

            num_hops_query = num_query_hops

            seen_eres = set()
            curr_eres = {ent_ere_id}

            curr_num_hops = 0

            stmt_id_sets = []

            while curr_num_hops < num_hops_query and len(curr_eres) > 0:
                seen_eres.update(curr_eres)

                stmt_ids = set.union(*[set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id]) for ere_id in curr_eres]) - set.union({target_event_stmt}, res_stmts, add_target_event_stmts, add_target_rel_stmts)

                if curr_num_hops > 0:
                    stmt_ids -= set.union(*stmt_id_sets[:curr_num_hops])


                for stmt_id in stmt_ids:
                    curr_eres.update([graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id])

                if stmt_ids:
                    stmt_id_sets.append(stmt_ids)

                curr_eres -= seen_eres
                curr_num_hops += 1

            query_stmts = set()

            if stmt_ids:
                reachable_eres = {ent_ere_id}

                min_num_query = min(5, len(set.union(*stmt_id_sets)))

                while len(query_stmts) < min_num_query:
                    curr_decay = 1

                    for hop_iter in range(len(stmt_id_sets)):
                        stmt_ids = stmt_id_sets[hop_iter] - set.union(query_stmts, {stmt_id for stmt_id in graph.stmts.keys() if graph.stmts[stmt_id].tail_id and not set.intersection(reachable_eres, {graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id})})

                        keep_list = np.random.random(len(stmt_ids))

                        stmt_ids_to_keep = set([item for iter, item in enumerate(stmt_ids) if keep_list[iter] < curr_decay])

                        query_stmts.update(stmt_ids_to_keep)

                        reachable_eres = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in query_stmts])

                        curr_decay *= decay_rate

                if query_stmts:
                    query_eres = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in query_stmts])

                    for ere_id in query_eres:
                        if graph.eres[ere_id].category == 'Relation':
                            query_stmts.update([item for item in graph.eres[ere_id].stmt_ids if graph.stmts[item].tail_id])

                    query_eres = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in query_stmts])
                    query_stmts.update(set.union(*[set([stmt_id for stmt_id in graph.eres[query_ere].stmt_ids if not graph.stmts[stmt_id].tail_id]) for query_ere in query_eres]))

            assert graph.eres[graph.stmts[target_event_stmt].head_id].category == 'Event'

            assert check_spans_distinct(graph, set(), res_stmts)

            for ere_id in orig_graph.eres.keys():
                orig_stmts = orig_graph.eres[ere_id].stmt_ids
                orig_neighs = orig_graph.eres[ere_id].neighbor_ere_ids

                if ere_id in graph.eres.keys():
                    if ere_id + '_dup' in graph.eres.keys():
                        assert len(set.intersection(graph.eres[ere_id].stmt_ids, orig_stmts)) == len(graph.eres[ere_id].stmt_ids)
                        assert len(set.intersection({item.split('_dup')[0] if not graph.stmts[item].tail_id else item for item in graph.eres[ere_id + '_dup'].stmt_ids}, orig_stmts)) == len(graph.eres[ere_id + '_dup'].stmt_ids)
                        assert not {stmt_id for stmt_id in set.intersection(graph.eres[ere_id].stmt_ids, {item.split('_dup')[0] if not graph.stmts[item].tail_id else item for item in graph.eres[ere_id + '_dup'].stmt_ids}) if graph.stmts[stmt_id].tail_id}
                        #assert len({item for item in graph.eres[ere_id].stmt_ids if graph.stmts[item].tail_id}) + len({item for item in graph.eres[ere_id + '_dup'].stmt_ids if graph.stmts[item].tail_id}) == len({item for item in orig_stmts if orig_graph.stmts[item].tail_id} - rej_span_stmts)
                        assert {item for item in graph.eres[ere_id].stmt_ids if not graph.stmts[item].tail_id} == {item.split('_dup')[0] for item in graph.eres[ere_id + '_dup'].stmt_ids if not graph.stmts[item].tail_id}
                        assert {item for item in graph.eres[ere_id].stmt_ids if not graph.stmts[item].tail_id} == {item for item in orig_stmts if not orig_graph.stmts[item].tail_id}
                        assert len(set.intersection({item.split('_dup')[0] for item in graph.eres[ere_id].neighbor_ere_ids}, orig_neighs)) == len(graph.eres[ere_id].neighbor_ere_ids)
                        assert len(set.intersection({item.split('_dup')[0] for item in graph.eres[ere_id + '_dup'].neighbor_ere_ids}, orig_neighs)) == len(graph.eres[ere_id + '_dup'].neighbor_ere_ids)
                    else:
                        assert len(set.intersection(graph.eres[ere_id].stmt_ids, orig_stmts)) == len(graph.eres[ere_id].stmt_ids)
                        assert {item for item in graph.eres[ere_id].stmt_ids if not graph.stmts[item].tail_id} == {item for item in orig_stmts if not orig_graph.stmts[item].tail_id}
                        assert len(set.intersection({item.split('_dup')[0] for item in graph.eres[ere_id].neighbor_ere_ids}, orig_neighs)) == len(graph.eres[ere_id].neighbor_ere_ids)

            res_stmts.update(set.union(*[{item for item in graph.eres[ere_id].stmt_ids if not graph.stmts[item].tail_id} for ere_id in res_eres]))
            res_stmts -= {item for item in graph.eres[ent_ere_id].stmt_ids if not graph.stmts[item].tail_id}

            assert not set.intersection({item.split('_dup')[0] for item in res_stmts if graph.stmts[item].tail_id}, {item.split('_dup')[0] for item in set(graph.stmts.keys()) - res_stmts if graph.stmts[item].tail_id})
            assert(len(set.intersection({item for item in graph.stmts.keys() if not graph.stmts[item].tail_id}, res_stmts)) == len(res_eres))

            cand_side_event_stmts = set([stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and stmt_id in res_stmts and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Event') and (len(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)])
            query_side_event_stmts = set([stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and stmt_id not in res_stmts and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Event') and (len(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)])

            if not cand_side_event_stmts:
                return False

            if (not allow_rel_query) and (not query_side_event_stmts):
                return False

            return graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt
        except:
            return False
    else:
        unused_second_ere_ids = set()

        while len(unused_second_ere_ids) == 0:
            ent_name = random.sample(list(set(entity_names.keys()) - tried_names), 1)[0]

            while (len(unused_second_ere_ids) == 0) and len(tried_ere_ids[ent_name]) < len(entity_names[ent_name]) - 1:
                ent_ere_id = random.sample(list(entity_names[ent_name] - tried_ere_ids[ent_name]), 1)[0]

                unused_second_ere_ids = []

                for ere_id in [sub_id for sub_id in (entity_names[ent_name] - tried_ere_ids[ent_name]) if set.intersection(entity_type_maps[ent_ere_id], entity_type_maps[sub_id])]:
                    if ere_id.split('_h')[0] != ent_ere_id.split('_h')[0] and frozenset([ent_ere_id, ere_id]) not in used_pairs:
                        graph_ere = first_graph_list[ere_id.split('_h')[0]]
                        graph_ent_ere = first_graph_list[ent_ere_id.split('_h')[0]]

                        event_statements = set.union(set([stmt_id for stmt_id in graph_ent_ere.eres[ent_ere_id].stmt_ids if graph_ent_ere.stmts[stmt_id].tail_id and (graph_ent_ere.eres[graph_ent_ere.stmts[stmt_id].head_id].category == 'Event') and (len(graph_ent_ere.eres[graph_ent_ere.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)]),
                                                     set([stmt_id for stmt_id in graph_ere.eres[ere_id].stmt_ids if graph_ere.stmts[stmt_id].tail_id and (graph_ere.eres[graph_ere.stmts[stmt_id].head_id].category == 'Event') and (len(graph_ere.eres[graph_ere.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)]))

                        if len(event_statements) < 1:
                            continue

                        ere_ev_labels = set()
                        ere_rel_labels = set()
                        ent_ere_ev_labels = set()
                        ent_ere_rel_labels = set()

                        if len([other_ere_id for other_ere_id in graph_ere.eres[ere_id].neighbor_ere_ids if graph_ere.eres[other_ere_id].category == 'Event']) > 0:
                            ere_ev_labels = set.union(*[set(graph_ere.eres[other_ere_id].label) for other_ere_id in graph_ere.eres[ere_id].neighbor_ere_ids if graph_ere.eres[other_ere_id].category == 'Event'])

                        if len([item for item in graph_ere.eres[ere_id].neighbor_ere_ids if (graph_ere.eres[item].category == 'Relation')]) > 0:
                            ere_rel_labels = set.union(*[set(graph_ere.eres[list(graph_ere.eres[item].neighbor_ere_ids - {ere_id})[0]].label) for item in graph_ere.eres[ere_id].neighbor_ere_ids if (graph_ere.eres[item].category == 'Relation')])

                        if len([other_ere_id for other_ere_id in graph_ent_ere.eres[ent_ere_id].neighbor_ere_ids if graph_ent_ere.eres[other_ere_id].category == 'Event']) > 0:
                            ent_ere_ev_labels = set.union(*[set(graph_ent_ere.eres[other_ere_id].label) for other_ere_id in graph_ent_ere.eres[ent_ere_id].neighbor_ere_ids if graph_ent_ere.eres[other_ere_id].category == 'Event'])

                        if len([item for item in graph_ent_ere.eres[ent_ere_id].neighbor_ere_ids if (graph_ent_ere.eres[item].category == 'Relation')]) > 0:
                            ent_ere_rel_labels = set.union(*[set(graph_ent_ere.eres[list(graph_ent_ere.eres[item].neighbor_ere_ids - {ent_ere_id})[0]].label) for item in graph_ent_ere.eres[ent_ere_id].neighbor_ere_ids if (graph_ent_ere.eres[item].category == 'Relation')])

                        if (not set.intersection(ere_ev_labels, ent_ere_ev_labels)) and (not set.intersection(ere_rel_labels, ent_ere_rel_labels)):
                            unused_second_ere_ids.append(ere_id)

                if len(unused_second_ere_ids) <= 1:
                    tried_ere_ids[ent_name].add(ent_ere_id)

                    if len(tried_ere_ids[ent_name]) == len(entity_names[ent_name]) - 1:
                        tried_names.add(ent_name)

        ere_id = random.sample(unused_second_ere_ids, 1)[0]
        graph_ere = first_graph_list[ere_id.split('_h')[0]]

        ent_ere_ids = {ent_ere_id, ere_id}

        event_statements_ent = set([stmt_id for stmt_id in graph_ent_ere.eres[ent_ere_id].stmt_ids if graph_ent_ere.stmts[stmt_id].tail_id and (graph_ent_ere.eres[graph_ent_ere.stmts[stmt_id].head_id].category == 'Event') and (len(graph_ent_ere.eres[graph_ent_ere.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)])
        event_statements_ere = set([stmt_id for stmt_id in graph_ere.eres[ere_id].stmt_ids if graph_ere.stmts[stmt_id].tail_id and (graph_ere.eres[graph_ere.stmts[stmt_id].head_id].category == 'Event') and (len(graph_ere.eres[graph_ere.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)])

        if len(event_statements_ent) > 0 and len(event_statements_ere) > 0:
            target_ere_id = random.sample(list(ent_ere_ids), 1)[0]
        elif len(event_statements_ent) > 0 and len(event_statements_ere) == 0:
            target_ere_id = ent_ere_id
        elif len(event_statements_ent) == 0 and len(event_statements_ere) > 0:
            target_ere_id = ere_id

        source_ere_id = list(ent_ere_ids - {target_ere_id})[0]

        source_graph_id = source_ere_id.split('_h')[0]
        target_graph_id = target_ere_id.split('_h')[0]

        source_graph = deepcopy(first_graph_list[source_graph_id])
        target_graph = deepcopy(first_graph_list[target_graph_id])

        target_ent = target_graph.eres[target_ere_id]

        target_event_stmts = set([stmt_id for stmt_id in target_ent.stmt_ids if target_graph.stmts[stmt_id].tail_id == target_ere_id and (target_graph.eres[target_graph.stmts[stmt_id].head_id].category == 'Event') and (len(target_graph.eres[target_graph.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)])

        target_event_stmt = random.sample(list(target_event_stmts), 1)[0]

        for ere_id in set(target_graph.eres.keys()) - {target_ere_id}:
            source_graph.eres[ere_id] = deepcopy(target_graph.eres[ere_id])

        for stmt_id in target_graph.stmts.keys():
            source_graph.stmts[stmt_id] = deepcopy(target_graph.stmts[stmt_id])

        for stmt_id in target_ent.stmt_ids:
            if source_graph.stmts[stmt_id].tail_id:
                source_graph.stmts[stmt_id].tail_id = source_ere_id
            else:
                source_graph.stmts[stmt_id].head_id = source_ere_id

        for ere_id in source_graph.eres.keys():
            if target_ere_id in source_graph.eres[ere_id].neighbor_ere_ids:
                source_graph.eres[ere_id].neighbor_ere_ids.discard(target_ere_id)
                source_graph.eres[ere_id].neighbor_ere_ids.add(source_ere_id)

        source_graph.eres[source_ere_id].stmt_ids.update(target_ent.stmt_ids)
        source_graph.eres[source_ere_id].neighbor_ere_ids = set.union(*[{source_graph.stmts[item].head_id, source_graph.stmts[item].tail_id} for item in source_graph.eres[source_ere_id].stmt_ids if source_graph.stmts[item].tail_id]) - {source_ere_id}

        for ere_id in source_graph.eres.keys():
            assert len(set.intersection(source_graph.eres[ere_id].neighbor_ere_ids, set(source_graph.eres.keys()))) == len(source_graph.eres[ere_id].neighbor_ere_ids)
            assert len(set.intersection({item for item in source_graph.eres[ere_id].stmt_ids if source_graph.stmts[item].tail_id}, set(source_graph.stmts.keys()))) == len({item for item in source_graph.eres[ere_id].stmt_ids if source_graph.stmts[item].tail_id})

        type_stmts = defaultdict(set)
        for (stmt_id, type_label) in [(stmt_id, (' ').join(source_graph.stmts[stmt_id].label)) for stmt_id in source_graph.eres[source_ere_id].stmt_ids if source_graph.stmts[stmt_id].tail_id is None]:
            type_stmts[type_label].add(stmt_id)

        for key in type_stmts.keys():
            if len(type_stmts[key]) > 1:
                if source_graph_id in [source_graph.stmts[stmt_id].graph_id for stmt_id in type_stmts[key]]:
                    for stmt_id in type_stmts[key]:
                        if source_graph.stmts[stmt_id].graph_id != source_graph_id:
                            source_graph.eres[source_ere_id].stmt_ids.discard(stmt_id)
                            del source_graph.stmts[stmt_id]
                else:
                    for stmt_id in list(type_stmts[key])[1:]:
                        source_graph.eres[source_ere_id].stmt_ids.discard(stmt_id)
                        del source_graph.stmts[stmt_id]

        seen_eres = set()
        curr_eres = {source_ere_id}

        i = 0

        while len(curr_eres) > 0 and i < 5:
            seen_eres.update(curr_eres)
            curr_eres = set.union(*[source_graph.eres[ere_id].neighbor_ere_ids for ere_id in curr_eres]) - seen_eres

            i += 1

        reachable_stmts = set.union(*[source_graph.eres[ere_id].stmt_ids for ere_id in seen_eres])
        seen_eres.update(curr_eres)

        for ere_id in deepcopy(seen_eres):
            if source_graph.eres[ere_id].category == 'Relation':
                reachable_stmts.update(source_graph.eres[ere_id].stmt_ids)
                seen_eres.update(source_graph.eres[ere_id].neighbor_ere_ids)

        reachable_stmts.update(set.union(*[{item for item in source_graph.eres[ere_id].stmt_ids if not source_graph.stmts[item].tail_id} for ere_id in seen_eres]))

        for ere_id in set(source_graph.eres.keys()) - seen_eres:
            del source_graph.eres[ere_id]

        for stmt_id in set(source_graph.stmts.keys()) - reachable_stmts:
            del source_graph.stmts[stmt_id]

        for ere_id in source_graph.eres.keys():
            source_graph.eres[ere_id].stmt_ids = set.intersection(source_graph.eres[ere_id].stmt_ids, reachable_stmts)
            source_graph.eres[ere_id].neighbor_ere_ids = set.intersection(source_graph.eres[ere_id].neighbor_ere_ids, seen_eres)#set.union(*[{source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id} for stmt_id in source_graph.eres[ere_id].stmt_ids if source_graph.stmts[stmt_id].tail_id])

        for ere_id in source_graph.eres.keys():
            assert len(set.intersection(source_graph.eres[ere_id].neighbor_ere_ids, set(source_graph.eres.keys()))) == len(source_graph.eres[ere_id].neighbor_ere_ids)
            assert len(set.intersection({item for item in source_graph.eres[ere_id].stmt_ids if source_graph.stmts[item].tail_id}, set(source_graph.stmts.keys()))) == len({item for item in source_graph.eres[ere_id].stmt_ids if source_graph.stmts[item].tail_id})

        num_hops_query = num_query_hops

        seen_eres = set()
        curr_eres = {source_ere_id}

        curr_num_hops = 0

        stmt_id_sets = []

        while curr_num_hops < num_hops_query and len(curr_eres) > 0:
            seen_eres.update(curr_eres)

            stmt_ids = set.union(*[set([stmt_id for stmt_id in source_graph.eres[ere_id].stmt_ids if source_graph.stmts[stmt_id].tail_id]) for ere_id in curr_eres]) - {stmt_id for stmt_id in source_graph.eres[source_ere_id].stmt_ids if source_graph.stmts[stmt_id].graph_id != source_graph_id}

            if curr_num_hops > 0:
                stmt_ids -= set.union(*stmt_id_sets[:curr_num_hops])

            #rel_stmts_to_add = set()

            for stmt_id in stmt_ids:
                # if source_graph.eres[source_graph.stmts[stmt_id].head_id].category == 'Relation':
                #     rel_stmts_to_add.update([item for item in source_graph.eres[source_graph.stmts[stmt_id].head_id].stmt_ids if source_graph.stmts[item].tail_id])
                #     curr_eres.update(source_graph.eres[source_graph.stmts[stmt_id].head_id].neighbor_ere_ids)

                curr_eres.update([source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id])

            #stmt_ids.update(rel_stmts_to_add)

            if stmt_ids:
                stmt_id_sets.append(stmt_ids)

            curr_eres -= seen_eres
            curr_num_hops += 1

        query_stmts = set()

        reachable_eres = {source_ere_id}

        min_num_query = min(5, len(set.union(*stmt_id_sets)))

        while len(query_stmts) < min_num_query:
            curr_decay = 1

            for hop_iter in range(len(stmt_id_sets)):
                stmt_ids = stmt_id_sets[hop_iter] - set.union(query_stmts, {stmt_id for stmt_id in source_graph.stmts.keys() if source_graph.stmts[stmt_id].tail_id and not set.intersection(reachable_eres, {source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id})})

                keep_list = np.random.random(len(stmt_ids))

                stmt_ids_to_keep = set([item for iter, item in enumerate(stmt_ids) if keep_list[iter] < curr_decay])

                # rel_stmts_to_add = set()
                # for stmt_id in stmt_ids_to_keep:
                #     if source_graph.eres[source_graph.stmts[stmt_id].head_id].category == 'Relation':
                #         rel_stmts_to_add.update([item for item in source_graph.eres[source_graph.stmts[stmt_id].head_id].stmt_ids if source_graph.stmts[item].tail_id])

                #stmt_ids_to_keep.update(rel_stmts_to_add)
                query_stmts.update(stmt_ids_to_keep)

                reachable_eres = set.union(*[{source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id} for stmt_id in query_stmts])

                curr_decay *= decay_rate

        query_eres = set.union(*[{source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id} for stmt_id in query_stmts])

        for ere_id in query_eres:
            if source_graph.eres[ere_id].category == 'Relation':
                query_stmts.update([item for item in source_graph.eres[ere_id].stmt_ids if source_graph.stmts[item].tail_id])

        query_eres = set.union(*[{source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id} for stmt_id in query_stmts])
        query_stmts.update(set.union(*[set([stmt_id for stmt_id in source_graph.eres[query_ere].stmt_ids if not source_graph.stmts[stmt_id].tail_id]) for query_ere in query_eres]))

        assert source_graph.eres[source_graph.stmts[target_event_stmt].head_id].category == 'Event'

        return source_graph, query_stmts, source_ere_id, ent_ere_ids, target_event_stmt

# This function pre-loads all json graphs in the given folder
def load_all_graphs_in_folder(graph_js_folder):
    print('Loading all graphs in {}...'.format(graph_js_folder))

    graph_file_list = sorted([f for f in Path(graph_js_folder).iterdir() if f.is_file()])

    graph_list = dict()

    for graph_file in tqdm(graph_file_list):
        if graph_file.is_file():
            graph = dill.load(open(graph_file, 'rb'))

            # rel_ere_map = defaultdict(set)
            # for ere_id in graph.eres.keys():
            #     if graph.eres[ere_id].category == 'Relation':
            #         rel_ere_map[frozenset(graph.eres[ere_id].neighbor_ere_ids)].add(ere_id)
            #
            # for key in rel_ere_map.keys():
            #     if len(rel_ere_map[key]) > 1:
            #         ere_id_to_keep = random.sample(list(rel_ere_map[key]), 1)[0]
            #
            #         for ere_id in rel_ere_map[key] - {ere_id_to_keep}:
            #             for stmt_id in {item for item in graph.eres[ere_id].stmt_ids if graph.stmts[item].tail_id}:
            #                 del graph.stmts[stmt_id]
            #
            #                 for neigh_ere_id in key:
            #                     graph.eres[neigh_ere_id].stmt_ids.discard(stmt_id)
            #
            #                     if {item for item in graph.eres[neigh_ere_id].stmt_ids if graph.stmts[item].tail_id}:
            #                         graph.eres[neigh_ere_id].neighbor_ere_ids = set.union(*[{graph.stmts[item].head_id, graph.stmts[item].tail_id} for item in graph.eres[neigh_ere_id].stmt_ids if graph.stmts[item].tail_id]) - {neigh_ere_id}
            #                     else:
            #                         del graph.eres[neigh_ere_id]
            #
            #             del graph.eres[ere_id]

            graph_list[str(graph_file).split('.p')[0].split('/')[-1]] = graph

            #graph =
    return graph_list


def filter_ent_names(entity_names, graph_list):
    keys_to_del = []

    for key in entity_names.keys():
        eres_to_dis = set()

        for ere_id in entity_names[key]:
            if ere_id.split('_h')[0] not in graph_list:
                eres_to_dis.add(ere_id)

        entity_names[key] -= eres_to_dis

        if len(entity_names[key]) == 0:
            keys_to_del.append(key)

    for key in keys_to_del:
        del entity_names[key]

def process_ent_names(entity_names, first_graph_list, allow_rel_query, one_step_connectedness_map, two_step_connectedness_map):
    self_pair_entity_names = set()

    keys_to_del = []

    for key in entity_names.keys():
        eres_to_dis = []

        for ere_id in entity_names[key]:
            skip = False

            graph = first_graph_list[ere_id.split('_h')[0]]

            event_stmts = set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Event') and (len(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)])
            rel_stmts = set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation') and (len(graph.eres[list(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids - {ere_id})[0]].neighbor_ere_ids) > 1)])

            if one_step_connectedness_map[ere_id] < 2 or two_step_connectedness_map[ere_id] < 4:
                eres_to_dis.append(ere_id)
                continue

            if not allow_rel_query:
                if len(event_stmts) == 0:
                    eres_to_dis.append(ere_id)
                    continue
            else:
                if len(event_stmts) == 0 and len(rel_stmts) == 0:
                    eres_to_dis.append(ere_id)
                    continue

            for stmt_id in graph.eres[ere_id].stmt_ids:
                if graph.stmts[stmt_id].tail_id:
                    for other_stmt_id in graph.eres[ere_id].stmt_ids - {stmt_id}:
                        if graph.stmts[other_stmt_id].tail_id:
                            if {graph.eres[graph.stmts[item].head_id].category for item in {stmt_id, other_stmt_id}} == {'Event'} and set.intersection(set(graph.eres[graph.stmts[other_stmt_id].head_id].label), set(graph.eres[graph.stmts[stmt_id].head_id].label)) or \
                                {graph.eres[graph.stmts[item].head_id].category for item in {stmt_id, other_stmt_id}} == {'Relation'} and set.intersection(set(graph.eres[list(graph.eres[graph.stmts[other_stmt_id].head_id].neighbor_ere_ids - {ere_id})[0]].label),
                                                                                                                                                           set(graph.eres[list(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids - {ere_id})[0]].label)):
                                skip = True
                                eres_to_dis.append(ere_id)
                                break
                if skip:
                    break

            if not allow_rel_query:
                if len(event_stmts) < 2:
                    continue
            else:
                if (len(event_stmts) == 0) or (len(event_stmts) + len(rel_stmts) < 2):
                    continue

            if (not skip) and one_step_connectedness_map[ere_id] >= 4 and two_step_connectedness_map[ere_id] >= 8:
                for stmt_1, stmt_2 in [item for item in itertools.combinations(list(set.union(event_stmts, rel_stmts)), 2) if 'Event' in {graph.eres[graph.stmts[sub_item].head_id].category for sub_item in item}]:
                    if {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Event'}:
                        inter_eres = set.intersection(graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids, graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids)
                        neighs_1 = graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids - inter_eres
                        neighs_2 = graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids - inter_eres

                        if len(inter_eres - {ere_id}) >= max(0, (1 - len(neighs_1))) + max(0, (1 - len(neighs_2))):
                            if check_distinct_ere_pair(graph, graph.stmts[stmt_1].head_id, graph.stmts[stmt_2].head_id):
                                self_pair_entity_names.add(ere_id)
                                break
                    elif allow_rel_query and {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Event', 'Relation'}:
                        if graph.eres[graph.stmts[stmt_2].head_id].category == 'Event':
                            temp = stmt_2
                            stmt_2 = stmt_1
                            stmt_1 = temp

                        rel_ere = graph.eres[graph.stmts[stmt_2].head_id]
                        event_neighs = graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids
                        other_rel_neigh = list(rel_ere.neighbor_ere_ids - {ere_id})[0]
                        rel_ev_neighs = {item for item in graph.eres[other_rel_neigh].neighbor_ere_ids if graph.eres[item].category == 'Event'}
                        rel_rel_neighs = set.union(*[(graph.eres[item].neighbor_ere_ids - {other_rel_neigh}) for item in graph.eres[other_rel_neigh].neighbor_ere_ids if graph.eres[item].category == 'Relation'])

                        if graph.stmts[stmt_1].head_id in rel_ev_neighs:
                            continue

                        rel_neighs = set.union(rel_ev_neighs, rel_rel_neighs)

                        inter_eres = set.intersection(event_neighs, rel_neighs)

                        neighs_1 = event_neighs - inter_eres
                        neighs_2 = rel_neighs - inter_eres

                        if len(inter_eres - {ere_id}) >= max(0, (1 - len(neighs_1))) + max(0, (1 - len(neighs_2))):
                            if check_distinct_ere_pair(graph, graph.stmts[stmt_1].head_id, graph.stmts[stmt_2].head_id):
                                self_pair_entity_names.add(ere_id)
                                break

        for ere_id in eres_to_dis:
            entity_names[key].discard(ere_id)

        if len(entity_names[key]) == 0:
            keys_to_del.append(key)

    for key in keys_to_del:
        del entity_names[key]

    return self_pair_entity_names

## This function generates graph mixtures from single-doc graphs chosen from the json folders it receives as parameters
#- If only first_js_folder is specified, make_mixture_data() will generate self-mixtures (i.e., it will mix only single-docs from first_js_folder)
#- If second_js_folder is also specified, make_mixture_data() will generate mixtures from single-docs taken from both first_js_folder and second_js_folder
def make_mixture_data(wiki_js_folder, event_names, entity_names, event_type_maps, entity_type_maps, one_step_connectedness_map, two_step_connectedness_map,
                      out_data_dir, num_sources, num_query_hops, query_decay, num_shared_eres, allow_rel_query, data_size, size_cut, print_every, min_connectedness_one_step, min_connectedness_two_step, perc_train, perc_val, perc_test):
    verify_dir(out_data_dir)
    verify_dir(os.path.join(out_data_dir, 'Train'))
    verify_dir(os.path.join(out_data_dir, 'Val'))
    verify_dir(os.path.join(out_data_dir, 'Test'))

    counter = 0

    graph_list = load_all_graphs_in_folder(wiki_js_folder)

    num_train_graphs = math.ceil(perc_train * len(graph_list))
    num_test_graphs = math.floor(perc_val * len(graph_list))

    unused_graphs = set(graph_list.keys())

    train_graph_list = set(random.sample(list(unused_graphs), num_train_graphs))
    unused_graphs -= train_graph_list
    test_graph_list = set(random.sample(list(unused_graphs), num_test_graphs))
    unused_graphs -= test_graph_list
    val_graph_list = unused_graphs

    dill.dump((val_graph_list, test_graph_list), open('/home/atomko/backup_drive/Summer_2020/gen_graph_lists.p', 'wb'))

    assert not set.intersection(train_graph_list, val_graph_list, test_graph_list)
    print(len(train_graph_list), len(val_graph_list), len(test_graph_list))

    train_entity_names = deepcopy(entity_names)
    val_entity_names = deepcopy(entity_names)
    test_entity_names = deepcopy(entity_names)

    filter_ent_names(train_entity_names, train_graph_list)
    filter_ent_names(val_entity_names, val_graph_list)
    filter_ent_names(test_entity_names, test_graph_list)

    train_self_pair_entity_names = process_ent_names(train_entity_names, graph_list, allow_rel_query, one_step_connectedness_map, two_step_connectedness_map)
    val_self_pair_entity_names = process_ent_names(val_entity_names, graph_list, allow_rel_query, one_step_connectedness_map, two_step_connectedness_map)
    test_self_pair_entity_names = process_ent_names(test_entity_names, graph_list, allow_rel_query, one_step_connectedness_map, two_step_connectedness_map)

    temp_1 = set.union(set.union(*[set([item.split('_h')[0] for item in train_entity_names[key]]) for key in train_entity_names.keys()]), set([item.split('_h')[0] for item in train_self_pair_entity_names]))
    temp_2 = set.union(set.union(*[set([item.split('_h')[0] for item in val_entity_names[key]]) for key in val_entity_names.keys()]), set([item.split('_h')[0] for item in val_self_pair_entity_names]))
    temp_3 = set.union(set.union(*[set([item.split('_h')[0] for item in test_entity_names[key]]) for key in test_entity_names.keys()]), set([item.split('_h')[0] for item in test_self_pair_entity_names]))

    print(len(temp_1), len(temp_2), len(temp_3))
    assert not set.intersection(temp_1, temp_2, temp_3)

    train_cut = perc_train * data_size
    val_cut = train_cut + (perc_val * data_size)

    used_pairs = set()

    pos = True

    tried_names = set()
    tried_ere_ids = defaultdict(set)

    entity_names = deepcopy(train_entity_names)
    self_pair_entity_names = deepcopy(train_self_pair_entity_names)

    while counter < data_size:
        if counter == train_cut:
            entity_names = deepcopy(val_entity_names)
            self_pair_entity_names = deepcopy(val_self_pair_entity_names)

            tried_names = set()
            tried_ere_ids = defaultdict(set)
        elif counter == val_cut:
            entity_names = deepcopy(test_entity_names)
            self_pair_entity_names = deepcopy(test_self_pair_entity_names)

            tried_names = set()
            tried_ere_ids = defaultdict(set)
        if pos:
            query_stmts = set()

            while not query_stmts:
                temp = create_mix(graph_list, event_names, entity_names, self_pair_entity_names, event_type_maps, entity_type_maps, allow_rel_query, one_step_connectedness_map, two_step_connectedness_map,
                                        num_sources, num_query_hops, query_decay, min_connectedness_one_step, min_connectedness_two_step, used_pairs, pos, tried_names, tried_ere_ids)

                if temp:
                    graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt = temp

                    # if len(dill.dumps(temp, -1)) >= 814433:
                    #     query_stmts = set()
                else:
                    query_stmts = set()

            self_pair_entity_names.discard(ent_ere_id)

            if counter < train_cut:
                dill.dump((graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt), open(os.path.join(out_data_dir, 'Train', re.sub('/', '_', re.sub('//', '_', ent_ere_id + '-self.p'))), 'wb'))
            elif counter < val_cut:
                dill.dump((graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt), open(os.path.join(out_data_dir, 'Val', re.sub('/', '_', re.sub('//', '_', ent_ere_id + '-self.p'))), 'wb'))
            else:
                dill.dump((graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt), open(os.path.join(out_data_dir, 'Test', re.sub('/', '_', re.sub('//', '_', ent_ere_id + '-self.p'))), 'wb'))
        else:
            size_check = False

            while not size_check:
                temp = create_mix(graph_list, event_names, entity_names, self_pair_entity_names, event_type_maps, entity_type_maps, allow_rel_query, one_step_connectedness_map,
                                        two_step_connectedness_map, num_sources, num_query_hops, query_decay, min_connectedness_one_step, min_connectedness_two_step, used_pairs, pos, tried_names, tried_ere_ids)

                source_graph, query_stmts, source_ere_id, ent_ere_ids, target_event_stmt = temp

                size_check = True

                # if len(dill.dumps(temp, -1)) < 814433:
                #     size_check = True

            used_pairs.add(frozenset(ent_ere_ids))

            if counter < train_cut:
                dill.dump((source_graph, query_stmts, source_ere_id, ent_ere_ids, target_event_stmt), open(os.path.join(out_data_dir, 'Train', re.sub('/', '_', re.sub('//', '_', '_'.join(list(ent_ere_ids)))) + '-pair.p'), 'wb'))
            elif counter < val_cut:
                dill.dump((source_graph, query_stmts, source_ere_id, ent_ere_ids, target_event_stmt), open(os.path.join(out_data_dir, 'Val', re.sub('/', '_', re.sub('//', '_', '_'.join(list(ent_ere_ids)))) + '-pair.p'), 'wb'))
            else:
                dill.dump((source_graph, query_stmts, source_ere_id, ent_ere_ids, target_event_stmt), open(os.path.join(out_data_dir, 'Test', re.sub('/', '_', re.sub('//', '_', '_'.join(list(ent_ere_ids)))) + '-pair.p'), 'wb'))

        pos = not pos

        print(counter)

        counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_data_dir", type=str, default="/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New", help='Folder where the mixtures will be written (will be created by the script, if it does not already exist)')
    parser.add_argument("--wiki_js_folder", type=str, default="/home/atomko/backup_drive/Summer_2020/Graph_Singles", help='First directory containing json graphs')
    parser.add_argument("--event_name_maps", type=str, default="/home/atomko/backup_drive/Summer_2020/event_names_uncollapsed.p", help='Name mapping for events')
    parser.add_argument("--entity_name_maps", type=str, default="/home/atomko/backup_drive/Summer_2020/entity_names_uncollapsed.p", help='Name mapping for entities')
    parser.add_argument("--event_type_maps", type=str, default="/home/atomko/backup_drive/Summer_2020/event_types_uncollapsed.p", help='Name mapping for events')
    parser.add_argument("--entity_type_maps", type=str, default="/home/atomko/backup_drive/Summer_2020/entity_types_uncollapsed.p", help='Name mapping for entities')
    parser.add_argument("--one_step_connectedness_map", type=str, default="/home/atomko/backup_drive/Summer_2020/connectedness_one_step_uncollapsed.p", help='Name mapping for events')
    parser.add_argument("--two_step_connectedness_map", type=str, default="/home/atomko/backup_drive/Summer_2020/connectedness_two_step_uncollapsed.p", help='Name mapping for entities')
    parser.add_argument("--num_sources", type=int, default=3, help='Number of single docs to mix at one time')
    parser.add_argument("--num_query_hops", type=int, default=4, help='Number of single docs to mix at one time')
    parser.add_argument("--query_decay", type=float, default=1, help='Number of mixtures to create')
    parser.add_argument("--num_shared_eres", type=int, default=3, help='Required number of mixture points in a produced mixture')
    parser.add_argument("--allow_rel_query", action='store_true', help='Required number of mixture points in a produced mixture')
    parser.add_argument("--data_size", type=int, default=1000, help='Number of mixtures to create')
    parser.add_argument("--size_cut", type=int, default=1500, help='Maximum number of EREs + statements for each mixture')
    parser.add_argument("--print_every", type=int, default=100, help='Print every x mixtures created')
    parser.add_argument("--min_connectedness_one_step", type=int, default=2, help='The minimum one-step connectedness score for an ERE to be selected as a mixture point')
    parser.add_argument("--min_connectedness_two_step", type=int, default=4, help='The minimum two-step connectedness score for an ERE to be selected as a mixture point')
    parser.add_argument("--perc_train", type=float, default=.8, help='Percentage of <data_size> mixtures to assign to the training set')
    parser.add_argument("--perc_val", type=float, default=.1, help='Percentage of <data_size> mixtures to assign to the validation set')
    parser.add_argument("--perc_test", type=float, default=.1, help='Percentage of <data_size> mixtures to assign to the test set')

    args = parser.parse_args()
    locals().update(vars(args))

    print("Params:\n", args, "\n")

    print("Generating mixtures ...\n")

    event_names = dill.load(open(event_name_maps, 'rb'))
    entity_names = dill.load(open(entity_name_maps, 'rb'))

    event_types = dill.load(open(event_type_maps, 'rb'))
    entity_types = dill.load(open(entity_type_maps, 'rb'))

    one_step_connectedness_map = dill.load(open(one_step_connectedness_map, 'rb'))
    two_step_connectedness_map = dill.load(open(two_step_connectedness_map, 'rb'))

    make_mixture_data(wiki_js_folder, event_names, entity_names, event_types, entity_types, one_step_connectedness_map, two_step_connectedness_map,
                      out_data_dir, num_sources, num_query_hops, query_decay, num_shared_eres, allow_rel_query, data_size, size_cut, print_every, min_connectedness_one_step, min_connectedness_two_step, perc_train, perc_val, perc_test)

    print("Done!\n")