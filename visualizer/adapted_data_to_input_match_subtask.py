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
from collections import defaultdict
from copy import deepcopy
from operator import itemgetter
from pathlib import Path

import dill
from tqdm import tqdm
import itertools
import numpy as np
from elmo_tokenize import get_rej_ere_ids

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
def create_mix(first_graph_list, second_graph_list, event_names, entity_names, self_pair_entity_names, event_type_maps, entity_type_maps, one_step_connectedness_map, two_step_connectedness_map,
                                        num_sources, num_query_hops, query_decay, min_connectedness_one_step, min_connectedness_two_step, used_pairs, pos, tried_names, tried_ere_ids):
    decay_rate = query_decay

    if pos:
        ent_ere_id = random.sample(list(self_pair_entity_names), 1)[0]
        graph = deepcopy(first_graph_list[ent_ere_id.split('_h')[0]])

        event_stmts = set([stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Event') and (len(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)])
        rel_stmts = set([stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation') and (len(graph.eres[list(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids - {ent_ere_id})[0]].neighbor_ere_ids) > 1)])

        poss_pairs = []

        for stmt_1, stmt_2 in [item for item in itertools.combinations(list(set.union(event_stmts, rel_stmts)), 2) if 'Event' in {graph.eres[graph.stmts[sub_item].head_id].category for sub_item in item}]:
            if {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Event'}:
                inter_eres = set.intersection(graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids, graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids)
                neighs_1 = graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids - inter_eres
                neighs_2 = graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids - inter_eres

                stmt_1_check = False
                stmt_2_check = False

                if len(inter_eres - {ent_ere_id}) >= max(0, (1 - len(neighs_1))) + max(0, (1 - len(neighs_2))):
                    same_head_label = set()
                    same_head_label.update([item for item in event_stmts if graph.eres[graph.stmts[item].head_id].category == 'Event' and set.intersection(set(graph.eres[graph.stmts[item].head_id].label), set(graph.eres[graph.stmts[stmt_1].head_id].label))])

                    if len(same_head_label) == 1:
                        stmt_1_check = True

                    same_head_label = set()
                    same_head_label.update([item for item in event_stmts if graph.eres[graph.stmts[item].head_id].category == 'Event' and set.intersection(set(graph.eres[graph.stmts[item].head_id].label), set(graph.eres[graph.stmts[stmt_2].head_id].label))])

                    if len(same_head_label) == 1:
                        stmt_2_check = True

                    if stmt_1_check and stmt_2_check:
                        poss_pairs.append((stmt_1, stmt_2, 2))
                    elif stmt_1_check:
                        poss_pairs.append((stmt_1, stmt_2, 0))
                    elif stmt_2_check:
                        poss_pairs.append((stmt_1, stmt_2, 1))
            elif {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Event', 'Relation'}:
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
                    same_head_label = set()
                    same_head_label.update([item for item in event_stmts if graph.eres[graph.stmts[item].head_id].category == 'Event' and set.intersection(set(graph.eres[graph.stmts[item].head_id].label), set(graph.eres[graph.stmts[stmt_1].head_id].label))])

                    if len(same_head_label) == 1:
                        poss_pairs.append((stmt_1, stmt_2, 0))
            # elif {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Relation'}:
            #     rel_ere_1 = graph.eres[graph.stmts[stmt_1].head_id]
            #     rel_ere_2 = graph.eres[graph.stmts[stmt_2].head_id]
            #     other_rel_neigh_1 = list(rel_ere_1.neighbor_ere_ids - {ent_ere_id})[0]
            #     other_rel_neigh_2 = list(rel_ere_2.neighbor_ere_ids - {ent_ere_id})[0]
            #     rel_ev_neighs_1 = {item for item in graph.eres[other_rel_neigh_1].neighbor_ere_ids if graph.eres[item].category == 'Event'}
            #     rel_rel_neighs_1 = set.union(*[(graph.eres[item].neighbor_ere_ids - {other_rel_neigh_1}) for item in graph.eres[other_rel_neigh_1].neighbor_ere_ids if graph.eres[item].category == 'Relation']) - {other_rel_neigh_2}
            #     rel_ev_neighs_2 = {item for item in graph.eres[other_rel_neigh_2].neighbor_ere_ids if graph.eres[item].category == 'Event'}
            #     rel_rel_neighs_2 = set.union(*[(graph.eres[item].neighbor_ere_ids - {other_rel_neigh_2}) for item in graph.eres[other_rel_neigh_2].neighbor_ere_ids if graph.eres[item].category == 'Relation']) - {other_rel_neigh_1}
            #
            #     inter_eres = set.intersection(set.union(rel_ev_neighs_1, rel_rel_neighs_1), set.union(rel_ev_neighs_2, rel_rel_neighs_2))
            #
            #     rel_neighs_1 = set.union(rel_ev_neighs_1, rel_rel_neighs_1) - inter_eres
            #     rel_neighs_2 = set.union(rel_ev_neighs_2, rel_rel_neighs_2) - inter_eres
            #
            #     stmt_1_check = False
            #     stmt_2_check = False
            #
            #     if len(inter_eres - {ent_ere_id}) >= max(0, (1 - len(rel_neighs_1))) + max(0, (1 - len(rel_neighs_2))):
            #         same_head_label = set()
            #         same_head_label.update([item for item in event_stmts if graph.eres[graph.stmts[item].head_id].category == 'Relation' and set.intersection(set(graph.eres[list(graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ent_ere_id})[0]].label),
            #                                                                                                                                                   set(graph.eres[list(graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids - {ent_ere_id})[0]].label))])
            #         if len(same_head_label) == 1:
            #             stmt_1_check = True
            #
            #         same_head_label = set()
            #         same_head_label.update([item for item in event_stmts if graph.eres[graph.stmts[item].head_id].category == 'Relation' and set.intersection(set(graph.eres[list(graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ent_ere_id})[0]].label),
            #                                                                                                                                                   set(graph.eres[list(graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids - {ent_ere_id})[0]].label))])
            #
            #         if len(same_head_label) == 1:
            #             stmt_2_check = True
            #
            #         if stmt_1_check and stmt_2_check:
            #             poss_pairs.append((stmt_1, stmt_2, 2))
            #         elif stmt_1_check:
            #             poss_pairs.append((stmt_1, stmt_2, 0))
            #         elif stmt_2_check:
            #             poss_pairs.append((stmt_1, stmt_2, 1))

        stmt_1, stmt_2, target_ind = random.sample(poss_pairs, 1)[0]

        if {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Event'}:
            inter_eres = set.intersection(graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids, graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids)
        elif {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Event', 'Relation'}:
            rel_ere = graph.eres[graph.stmts[stmt_2].head_id]
            event_neighs = graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids
            other_rel_neigh = list(rel_ere.neighbor_ere_ids - {ent_ere_id})[0]
            rel_ev_neighs = {item for item in graph.eres[other_rel_neigh].neighbor_ere_ids if graph.eres[item].category == 'Event'}
            rel_rel_neighs = set.union(*[(graph.eres[item].neighbor_ere_ids - {other_rel_neigh}) for item in graph.eres[other_rel_neigh].neighbor_ere_ids if graph.eres[item].category == 'Relation'])

            rel_neighs = set.union(rel_ev_neighs, rel_rel_neighs)

            inter_eres = set.intersection(event_neighs, rel_neighs)

            neighs_2 = rel_neighs - inter_eres
        # elif {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Relation'}:
        #     rel_ere_1 = graph.eres[graph.stmts[stmt_1].head_id]
        #     rel_ere_2 = graph.eres[graph.stmts[stmt_2].head_id]
        #     other_rel_neigh_1 = list(rel_ere_1.neighbor_ere_ids - {ent_ere_id})[0]
        #     other_rel_neigh_2 = list(rel_ere_2.neighbor_ere_ids - {ent_ere_id})[0]
        #     rel_ev_neighs_1 = {item for item in graph.eres[other_rel_neigh_1].neighbor_ere_ids if graph.eres[item].category == 'Event'}
        #     rel_rel_neighs_1 = set.union(*[(graph.eres[item].neighbor_ere_ids - {other_rel_neigh_1}) for item in graph.eres[other_rel_neigh_1].neighbor_ere_ids if graph.eres[item].category == 'Relation']) - {other_rel_neigh_2}
        #     rel_ev_neighs_2 = {item for item in graph.eres[other_rel_neigh_2].neighbor_ere_ids if graph.eres[item].category == 'Event'}
        #     rel_rel_neighs_2 = set.union(*[(graph.eres[item].neighbor_ere_ids - {other_rel_neigh_2}) for item in graph.eres[other_rel_neigh_2].neighbor_ere_ids if graph.eres[item].category == 'Relation']) - {other_rel_neigh_1}
        #
        #     inter_eres = set.intersection(set.union(rel_ev_neighs_1, rel_rel_neighs_1), set.union(rel_ev_neighs_2, rel_rel_neighs_2))
        #
        #     neighs_1 = set.union(rel_ev_neighs_1, rel_rel_neighs_1) - inter_eres
        #     neighs_2 = set.union(rel_ev_neighs_2, rel_rel_neighs_2) - inter_eres

        if target_ind == 2:
            target_event_stmt = random.sample([stmt_1, stmt_2], 1)[0]
        elif target_ind == 0:
            target_event_stmt = stmt_1
        elif target_ind == 1:
            target_event_stmt = stmt_2

        if graph.eres[graph.stmts[target_event_stmt].head_id].category == 'Event':
            target_event_ere_id = graph.stmts[target_event_stmt].head_id
        # elif graph.eres[graph.stmts[target_event_stmt].head_id].category == 'Relation':
        #     target_event_ere_id = list(graph.eres[graph.stmts[target_event_stmt].head_id].neighbor_ere_ids - {ent_ere_id})[0]

        other_event_stmt = list({stmt_1, stmt_2} - {target_event_stmt})[0]

        if graph.eres[graph.stmts[other_event_stmt].head_id].category == 'Event':
            other_event_ere_id = graph.stmts[other_event_stmt].head_id
        elif graph.eres[graph.stmts[other_event_stmt].head_id].category == 'Relation':
            other_event_ere_id = list(graph.eres[graph.stmts[other_event_stmt].head_id].neighbor_ere_ids - {ent_ere_id})[0]


        if graph.eres[graph.stmts[target_event_stmt].head_id].category == 'Event':
            if graph.eres[graph.stmts[other_event_stmt].head_id].category == 'Event':
                neighs_other_event = graph.eres[other_event_ere_id].neighbor_ere_ids - inter_eres
            elif graph.eres[graph.stmts[other_event_stmt].head_id].category == 'Relation':
                neighs_other_event = neighs_2

            neighs_target_event = graph.eres[target_event_ere_id].neighbor_ere_ids - inter_eres

            source_eres = set()
            target_eres = set()

            ere_pool = deepcopy(inter_eres - {ent_ere_id})

            if len(neighs_other_event) > len(neighs_target_event):
                target_eres.update(set(random.sample(list(ere_pool), min(len(neighs_other_event) - len(neighs_target_event), len(ere_pool)))))
            elif len(neighs_target_event) > len(neighs_other_event):
                source_eres.update(set(random.sample(list(ere_pool), min(len(neighs_target_event) - len(neighs_other_event), len(ere_pool)))))

            ere_pool -= set.union(target_eres, source_eres)

            target = True

            while len(ere_pool) > 0:
                if target:
                    target_eres.update(set(random.sample(list(ere_pool), 1)))
                    target = False
                else:
                    source_eres.update(set(random.sample(list(ere_pool), 1)))
                    target = True

                ere_pool -= set.union(target_eres, source_eres)

        #print(len(neighs_other_event), len(source_eres), len(neighs_target_event), len(target_eres), len(inter_eres - {ent_ere_id}))
            #target_eres = set(random.sample(list(inter_eres - {ent_ere_id}), math.ceil(float(len(inter_eres - {ent_ere_id})) / 2)))
        # elif graph.eres[graph.stmts[target_event_stmt].head_id].category == 'Relation':
        #     if target_event_stmt == stmt_1:
        #         target_eres = set(random.sample(list(inter_eres - {ent_ere_id}), max(0, (1 - len(neighs_1)))))
        #     elif target_event_stmt == stmt_2:
        #         target_eres = set(random.sample(list(inter_eres - {ent_ere_id}), max(0, (1 - len(neighs_2)))))

        source_res_eres = {other_event_ere_id}
        target_res_eres = {target_event_ere_id}

        other_eres = {item for item in graph.eres[ent_ere_id].neighbor_ere_ids if item not in {target_event_ere_id, graph.stmts[other_event_stmt].head_id}}

        source = True

        while len(other_eres) > 0:
            if source:
                source_res_eres.update(set(random.sample(list(other_eres), 1)))
                source = False
            else:
                target_res_eres.update(set(random.sample(list(other_eres), 1)))
                source = True

            other_eres -= set.union(source_res_eres, target_res_eres)

        # For candidate side
        seen_eres_target = set.union(source_res_eres, source_eres, neighs_other_event)
        # if other_event_stmt == stmt_1:
        #     seen_eres = set.union((inter_eres - target_eres), {other_event_ere_id}, neighs_1)
        #     other_event_neighs = neighs_1
        # elif other_event_stmt == stmt_2:

        # For query side
        if graph.eres[graph.stmts[target_event_stmt].head_id].category == 'Event':
            seen_eres_other = set.union(target_res_eres, target_eres, neighs_target_event)
        # elif graph.eres[graph.stmts[other_event_stmt].head_id].category == 'Relation':
        #     # if other_event_stmt == stmt_1:
        #     #     seen_eres = set.union((inter_eres - target_eres), {other_event_ere_id}, neighs_1)
        #     #     other_event_neighs = neighs_1
        #     # elif other_event_stmt == stmt_2:
        #     other_event_neighs = neighs_2
        #     seen_eres = set.union(source_res_eres, source_eres, other_event_neighs)

        curr_eres_target = {ent_ere_id}
        curr_eres_other = {ent_ere_id}
        curr_num_hops = 0

        res_stmts_target = set()
        res_stmts_other = set()
        
        while len(curr_eres_target) > 0 and len(curr_eres_other) > 0:
            # For candidate side
            seen_eres_target.update(curr_eres_target)

            stmt_ids_target = set.union(*[set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id]) for ere_id in curr_eres_target])
            print([(graph.eres[graph.stmts[item].head_id].label, graph.stmts[item].label, graph.eres[graph.stmts[item].tail_id].label) for item in stmt_ids_target])
            print([graph.eres[item].label for item in set.union(source_res_eres, source_eres, neighs_other_event, (seen_eres_other - {ent_ere_id}))])

            for stmt_id in stmt_ids_target:
                if not set.intersection(set.union(source_res_eres, source_eres, neighs_other_event, (seen_eres_other - {ent_ere_id})), {graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id}):
                    if graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
                        if not set.intersection(set.union(*[{graph.stmts[neigh_stmt_id].head_id, graph.stmts[neigh_stmt_id].tail_id} for neigh_stmt_id in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[neigh_stmt_id].tail_id]), \
                                                set.union(source_res_eres, source_eres, neighs_other_event, (seen_eres_other - {ent_ere_id}))):
                            res_stmts_target.add(stmt_id)
                    else:
                        res_stmts_target.add(stmt_id)
            assert not set.intersection(set.union(source_res_eres, source_eres, neighs_other_event, (seen_eres_other - {ent_ere_id})), set.union(target_res_eres, target_eres, neighs_target_event, (seen_eres_target - {ent_ere_id})))
            print(graph.graph_id, graph.eres[ent_ere_id].label)
            print([graph.eres[item].label for item in source_res_eres])
            print([graph.eres[item].label for item in source_eres])
            print([graph.eres[item].label for item in neighs_other_event])
            print([graph.eres[item].label for item in target_res_eres])
            print([graph.eres[item].label for item in target_eres])
            print([graph.eres[item].label for item in neighs_target_event])

            curr_eres_target = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in res_stmts_target if graph.stmts[stmt_id].tail_id]) - seen_eres_target
            seen_eres_other = set.union(seen_eres_other, deepcopy(curr_eres_target))

            seen_eres_other.update(curr_eres_other)

            stmt_ids_other = set.union(*[set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id]) for ere_id in curr_eres_other])

            for stmt_id in stmt_ids_other:
                if not set.intersection(set.union(target_res_eres, target_eres, neighs_target_event, (seen_eres_target - {ent_ere_id})), {graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id}):
                    if graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
                        if not set.intersection(set.union(*[{graph.stmts[neigh_stmt_id].head_id, graph.stmts[neigh_stmt_id].tail_id} for neigh_stmt_id in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[neigh_stmt_id].tail_id]), \
                                                set.union(target_res_eres, target_eres, neighs_target_event, (seen_eres_target - {ent_ere_id}))):
                            res_stmts_other.add(stmt_id)
                    else:
                        res_stmts_other.add(stmt_id)

            curr_eres_other = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in res_stmts_other if graph.stmts[stmt_id].tail_id]) - seen_eres_other
            seen_eres_target = set.union(seen_eres_target, deepcopy(curr_eres_other))

            curr_num_hops += 1


        # if graph.eres[graph.stmts[target_event_stmt].head_id].category == 'Event':
        #     res_stmts.add(target_event_stmt)
        # elif graph.eres[graph.stmts[target_event_stmt].head_id].category == 'Relation':
        #     res_stmts.update({item for item in graph.eres[graph.stmts[target_event_stmt].head_id].stmt_ids if graph.stmts[item].tail_id})

        res_eres_target = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in res_stmts_target if graph.stmts[stmt_id].tail_id]) - {ent_ere_id}
        res_eres_other = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in res_stmts_other if graph.stmts[stmt_id].tail_id]) - {ent_ere_id}

        # for res_ere in res_eres:
        #     if graph.eres[res_ere].category == 'Relation':
        #         res_stmts.update([item for item in graph.eres[res_ere].stmt_ids if graph.stmts[item].tail_id])

        #res_eres = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} for stmt_id in res_stmts if graph.stmts[stmt_id].tail_id]) - {ent_ere_id}

        dill.dump((graph, res_stmts_target, res_eres_target, res_stmts_other, res_eres_other), open('/home/atomko/backup_drive/Summer_2020/res_test.p', 'wb'))

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

        rej_ere_ids = get_rej_ere_ids(graph, res_stmts)

        if len(rej_ere_ids) > 0:
            rej_span_stmts = set.union(*[graph.eres[rej_ere_id].stmt_ids for rej_ere_id in rej_ere_ids])
        else:
            rej_span_stmts = set()

        seen_eres = set()
        curr_eres = {ent_ere_id}

        while len(curr_eres) > 0:
            seen_eres.update(curr_eres)

            curr_eres = set.union(*[graph.eres[ere_id].neighbor_ere_ids for ere_id in curr_eres]) - set.union(seen_eres, rej_ere_ids)

        reachable_stmts = set.union(*[graph.eres[ere_id].stmt_ids for ere_id in seen_eres]) - rej_span_stmts

        for ere_id in set(graph.eres.keys()) - seen_eres:
            del graph.eres[ere_id]

        for stmt_id in set(graph.stmts.keys()) - reachable_stmts:
            del graph.stmts[stmt_id]

        for ere_id in set(graph.eres.keys()):
            graph.eres[ere_id].stmt_ids -= rej_span_stmts
            graph.eres[ere_id].neighbor_ere_ids -= rej_ere_ids

        for res_ere in res_eres:
             assert len(set.intersection({item for item in graph.eres[res_ere].stmt_ids if graph.stmts[item].tail_id}, res_stmts)) == len({item for item in graph.eres[res_ere].stmt_ids if graph.stmts[item].tail_id})

        num_hops_query = num_query_hops

        seen_eres = set()
        curr_eres = {ent_ere_id}

        curr_num_hops = 0

        stmt_id_sets = []

        while curr_num_hops < num_hops_query and len(curr_eres) > 0:
            seen_eres.update(curr_eres)

            stmt_ids = set.union(*[set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id]) for ere_id in curr_eres]) - {target_event_stmt}

            if curr_num_hops > 0:
                stmt_ids -= set.union(*stmt_id_sets[:curr_num_hops])

            #rel_stmts_to_add = set()

            for stmt_id in stmt_ids:
                # if graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
                #     rel_stmts_to_add.update([item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id])
                #     curr_eres.update(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids)

                curr_eres.update([graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id])

            #stmt_ids.update(rel_stmts_to_add)

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

                    #rel_stmts_to_add = set()
                    # for stmt_id in stmt_ids_to_keep:
                    #     if graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
                    #         rel_stmts_to_add.update([item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id])

                    #stmt_ids_to_keep.update(rel_stmts_to_add)
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

        return graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt
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
            source_ere_id = random.sample(list(ent_ere_ids), 1)[0]
        elif len(event_statements_ent) > 0 and len(event_statements_ere) == 0:
            source_ere_id = ere_id
        elif len(event_statements_ent) == 0 and len(event_statements_ere) > 0:
            source_ere_id = ent_ere_id

        target_ere_id = list(ent_ere_ids - {source_ere_id})[0]

        source_graph_id = source_ere_id.split('_h')[0]
        target_graph_id = target_ere_id.split('_h')[0]

        source_graph = deepcopy(first_graph_list[source_graph_id])
        target_graph = deepcopy(first_graph_list[target_graph_id])

        target_ent = target_graph.eres[target_ere_id]

        target_event_stmts = set([stmt_id for stmt_id in target_ent.stmt_ids if target_graph.stmts[stmt_id].tail_id and (target_graph.eres[target_graph.stmts[stmt_id].head_id].category == 'Event') and (len(target_graph.eres[target_graph.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)])
                                       # set([stmt_id for stmt_id in target_ent.stmt_ids if target_graph.stmts[stmt_id].tail_id == target_ere_id and (target_graph.eres[target_graph.stmts[stmt_id].head_id].category == 'Relation') and \
                                       #     (len(target_graph.eres[list(target_graph.eres[target_graph.stmts[stmt_id].head_id].neighbor_ere_ids - {target_ere_id})[0]].neighbor_ere_ids) > 1)]))

        target_event_stmt = random.sample(target_event_stmts, 1)[0]

        # if target_graph.eres[target_graph.stmts[target_event_stmt].head_id].category == 'Event':
        #     target_event_ere_id = target_graph.stmts[target_event_stmt].head_id
        #     source_graph.eres[target_event_ere_id] = deepcopy(target_graph.eres[target_event_ere_id])
        #     source_graph.eres[target_event_ere_id].neighbor_ere_ids = set.union({source_ere_id}, (target_graph.eres[target_event_ere_id].neighbor_ere_ids - {target_ere_id}))
        #     source_graph.stmts.update({stmt_id: deepcopy(target_graph.stmts[stmt_id]) for stmt_id in target_graph.eres[target_event_ere_id].stmt_ids if graph.stmts[stmt_id].tail_id})
        # elif target_graph.eres[target_graph.stmts[target_event_stmt].head_id].category == 'Relation':
        #     target_event_ere_id = list(target_graph.eres[target_graph.stmts[target_event_stmt].head_id].neighbor_ere_ids - {target_ere_id})[0]
        #     source_graph.eres[target_graph.stmts[target_event_stmt].head_id] = deepcopy(target_graph.eres[target_graph.stmts[target_event_stmt].head_id])
        #     source_graph.eres[target_event_ere_id] = deepcopy(target_graph.eres[target_event_ere_id])
        #     source_graph.eres[target_graph.stmts[target_event_stmt].head_id].neighbor_ere_ids = set.union({source_ere_id}, (target_graph.eres[target_graph.stmts[target_event_stmt].head_id].neighbor_ere_ids - {target_ere_id}))
        #     source_graph.stmts.update({stmt_id: deepcopy(target_graph.stmts[stmt_id]) for stmt_id in target_graph.eres[target_graph.stmts[target_event_stmt].head_id].stmt_ids})
        #
        #     rem_stmt_ids = set()
        #
        #     for stmt_id in target_graph.eres[target_event_ere_id].stmt_ids:
        #         if target_graph.eres[target_graph.stmts[stmt_id].head_id].category == 'Relation' and set.intersection(target_graph.eres[target_graph.stmts[stmt_id].head_id].neighbor_ere_ids, {target_ere_id}) and target_graph.stmts[stmt_id].head_id != target_graph.stmts[target_event_stmt].head_id:
        #             rem_stmt_ids.add(stmt_id)
        #
        #     for stmt_id in (target_graph.eres[target_event_ere_id].stmt_ids - rem_stmt_ids):
        #         source_graph.stmts.update({stmt_id: deepcopy(target_graph.stmts[stmt_id])})
        #
        #     source_graph.eres[target_event_ere_id].stmt_ids -= rem_stmt_ids
        #     source_graph.eres[target_event_ere_id].neighbor_ere_ids = {source_graph.stmts[item].head_id for item in source_graph.eres[target_event_ere_id].stmt_ids if source_graph.stmts[item].tail_id}

        # if target_graph.eres[target_graph.stmts[target_event_stmt].head_id].category == 'Event':
        #     seen_eres = {target_ere_id}
        # elif target_graph.eres[target_graph.stmts[target_event_stmt].head_id].category == 'Relation':
        #     seen_eres = {target_graph.stmts[target_event_stmt].head_id, target_ere_id}

        curr_eres = [target_ere_id]
        seen_eres = set()

        curr_num_hops = 0

        while (len(curr_eres) > 0): # and (curr_num_hops <= 5):
            seen_eres.update(curr_eres)
            next_eres = set([item for neighs in [target_graph.eres[ere_id].neighbor_ere_ids for ere_id in curr_eres] for item in neighs]) - seen_eres

            # for next_ere in deepcopy(next_eres):
                # if target_graph.eres[next_ere].category == 'Relation':
                #     if not set.intersection(target_graph.eres[next_ere].neighbor_ere_ids, {target_ere_id}):
                #       next_eres.add(next_ere)
                    # else:
                    #     next_eres.discard(next_ere)

            next_eres -= seen_eres

            curr_eres = list(next_eres)

            curr_num_hops += 1

        eres_to_add = set()

        for ere_id in seen_eres:
            if target_graph.eres[ere_id].category == 'Relation':
                eres_to_add.update(target_graph.eres[ere_id].neighbor_ere_ids)

        seen_eres.update(eres_to_add)
        seen_eres -= {target_ere_id}
        source_graph.eres.update({ere_id: deepcopy(target_graph.eres[ere_id]) for ere_id in seen_eres})

        #target_ere_rel_neighs = {sub_item for sub_item in target_graph.eres[target_ere_id].neighbor_ere_ids if target_graph.eres[sub_item].category == 'Relation'}

        for ere_id in seen_eres:
            # neighs_to_discard = set()
            #
            # for item in source_graph.eres[ere_id].neighbor_ere_ids:
            #     if item == target_ere_id or item in target_ere_rel_neighs:
            #         neighs_to_discard.add(item)
            #
            # for item in neighs_to_discard:
            #     source_graph.eres[ere_id].neighbor_ere_ids.discard(item)
            #
            # source_graph.eres[ere_id].neighbor_ere_ids = set([item for item in source_graph.eres[ere_id].neighbor_ere_ids if item in seen_eres])
            # stmt_ids_to_dis = set()
            #
            # for stmt_id in [item for item in target_graph.eres[ere_id].stmt_ids if target_graph.stmts[item].tail_id]:
            #     head_tail = {target_graph.stmts[stmt_id].head_id, target_graph.stmts[stmt_id].tail_id}
            #
            #     if (len(set.intersection(head_tail, seen_eres)) < 2) or (target_ere_id in head_tail) or set.intersection(target_ere_rel_neighs, head_tail):
            #         stmt_ids_to_dis.add(stmt_id)
            #
            # source_graph.eres[ere_id].stmt_ids -= stmt_ids_to_dis
            if ere_id in target_ent.neighbor_ere_ids:
                source_graph.eres[ere_id].neighbor_ere_ids = set.union((source_graph.eres[ere_id].neighbor_ere_ids - {target_ere_id}), {source_ere_id})

        source_graph.stmts.update({stmt_id: deepcopy(stmt) for stmt_id, stmt in target_graph.stmts.items() if (len(set.intersection({stmt.head_id, stmt.tail_id}, seen_eres)) == 2) or (stmt.head_id in seen_eres and not stmt.tail_id)})

        source_graph.stmts.update({stmt_id: deepcopy(target_graph.stmts[stmt_id]) for stmt_id in target_graph.eres[target_ere_id].stmt_ids})
        source_graph.eres[source_ere_id].stmt_ids.update(target_graph.eres[target_ere_id].stmt_ids)
        source_graph.eres[source_ere_id].neighbor_ere_ids.update({target_graph.stmts[item].head_id for item in target_graph.eres[target_ere_id].stmt_ids if target_graph.stmts[item].tail_id})

        for stmt_id in source_graph.eres[source_ere_id].stmt_ids:
            if source_graph.stmts[stmt_id].head_id == target_ere_id:
                source_graph.stmts[stmt_id].head_id = source_ere_id
            elif source_graph.stmts[stmt_id].tail_id == target_ere_id:
                source_graph.stmts[stmt_id].tail_id = source_ere_id

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


        # if source_graph.eres[source_graph.stmts[target_event_stmt].head_id].category == 'Event':
        #     seen_eres = {source_ere_id}
        # elif source_graph.eres[source_graph.stmts[target_event_stmt].head_id].category == 'Relation':
        #     seen_eres = {source_graph.stmts[target_event_stmt].head_id}

        curr_eres = [source_ere_id]

        while len(curr_eres) > 0:
            seen_eres.update(curr_eres)
            #print(source_graph_id, target_graph_id, target_ere_id)
            next_eres = set([item for neighs in [source_graph.eres[ere_id].neighbor_ere_ids for ere_id in curr_eres] for item in neighs]) - seen_eres

            curr_eres = list(next_eres)

        if source_graph.eres[source_graph.stmts[target_event_stmt].head_id].category == 'Event':
            reachable_source_eres = seen_eres
        # elif source_graph.eres[source_graph.stmts[target_event_stmt].head_id].category == 'Relation':
        #     reachable_source_eres = seen_eres - {source_graph.stmts[target_event_stmt].head_id}

        reachable_source_stmts = {stmt_id for stmt_id in source_graph.stmts.keys() if set.intersection({source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id}, reachable_source_eres)}

        for ere_id in set(source_graph.eres.keys()) - reachable_source_eres:
            if source_graph.eres[ere_id].graph_id == source_graph_id:
                del source_graph.eres[ere_id]

        for stmt_id in set(source_graph.stmts.keys()) - reachable_source_stmts:
            if source_graph.stmts[stmt_id].graph_id == source_graph_id:
                del source_graph.stmts[stmt_id]

        num_hops_query = num_query_hops

        seen_eres = set()
        curr_eres = {source_ere_id}

        curr_num_hops = 0

        stmt_id_sets = []

        while curr_num_hops < num_hops_query and len(curr_eres) > 0:
            seen_eres.update(curr_eres)

            stmt_ids = set.union(*[set([stmt_id for stmt_id in source_graph.eres[ere_id].stmt_ids if source_graph.stmts[stmt_id].tail_id]) for ere_id in curr_eres]) - {item for item in source_graph.eres[source_ere_id].stmt_ids if source_graph.stmts[item].graph_id == target_graph_id}

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

        query_stmts = set.union(*stmt_id_sets)
########################################################
        # query_stmts = set()

        # reachable_eres = {source_ere_id}
        # min_num_query = min(5, len(set.union(*stmt_id_sets)))
        #
        # while len(query_stmts) < min_num_query:
        #     curr_decay = 1
        #
        #     for hop_iter in range(len(stmt_id_sets)):
        #         stmt_ids = stmt_id_sets[hop_iter] - set.union(query_stmts, {stmt_id for stmt_id in source_graph.stmts.keys() if source_graph.stmts[stmt_id].tail_id and not set.intersection(reachable_eres, {source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id})})
        #
        #         keep_list = np.random.random(len(stmt_ids))
        #
        #         stmt_ids_to_keep = set([item for iter, item in enumerate(stmt_ids) if keep_list[iter] < curr_decay])
        #
        #         # rel_stmts_to_add = set()
        #         # for stmt_id in stmt_ids_to_keep:
        #         #     if source_graph.eres[source_graph.stmts[stmt_id].head_id].category == 'Relation':
        #         #         rel_stmts_to_add.update([item for item in source_graph.eres[source_graph.stmts[stmt_id].head_id].stmt_ids if source_graph.stmts[item].tail_id])
        #
        #         #stmt_ids_to_keep.update(rel_stmts_to_add)
        #         query_stmts.update(stmt_ids_to_keep)
        #
        #         reachable_eres = set.union(*[{source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id} for stmt_id in query_stmts])
        #
        #         curr_decay *= decay_rate
        #
        #         if len(query_stmts) >= min_num_query:
        #             break
#############################################################
        query_eres = set.union(*[{source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id} for stmt_id in query_stmts])

        # for ere_id in query_eres:
        #     if source_graph.eres[ere_id].category == 'Relation':
        #         query_stmts.update([item for item in source_graph.eres[ere_id].stmt_ids if source_graph.stmts[item].tail_id])

        # query_eres = set.union(*[{source_graph.stmts[stmt_id].head_id, source_graph.stmts[stmt_id].tail_id} for stmt_id in query_stmts])
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
            graph_list[str(graph_file).split('.p')[0].split('/')[-1]] = dill.load(open(graph_file, 'rb'))

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

def process_ent_names(entity_names, first_graph_list, one_step_connectedness_map, two_step_connectedness_map):
    self_pair_entity_names = set()

    keys_to_del = []

    for key in entity_names.keys():
        eres_to_dis = []

        for ere_id in entity_names[key]:
            skip = False

            graph = first_graph_list[ere_id.split('_h')[0]]

            event_stmts = set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Event') and (len(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids) > 1)])
            rel_stmts = set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation') and (len(graph.eres[list(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids - {ere_id})[0]].neighbor_ere_ids) > 1)])

            if len(event_stmts) == 0 or one_step_connectedness_map[ere_id] < 4:
                eres_to_dis.append(ere_id)
                skip = True

            seen_head_tail = set()

            for stmt_id in graph.eres[ere_id].stmt_ids:
                if graph.stmts[stmt_id].tail_id:
                    if (graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id) in seen_head_tail:
                        eres_to_dis.append(ere_id)
                        skip = True
                        break
                    else:
                        seen_head_tail.add((graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id))

            if (not skip) and (len(set.union(event_stmts, rel_stmts)) >= 2) and (one_step_connectedness_map[ere_id] >= 8):
                for stmt_1, stmt_2 in [item for item in itertools.combinations(list(set.union(event_stmts, rel_stmts)), 2) if 'Event' in {graph.eres[graph.stmts[sub_item].head_id].category for sub_item in item}]:
                    if {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Event'}:
                        inter_eres = set.intersection(graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids, graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids)
                        neighs_1 = graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids - inter_eres
                        neighs_2 = graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids - inter_eres

                        if len(inter_eres - {ere_id}) >= max(0, (1 - len(neighs_1))) + max(0, (1 - len(neighs_2))):
                            same_head_label = set()
                            same_head_label.update([item for item in event_stmts if graph.eres[graph.stmts[item].head_id].category == 'Event' and set.intersection(set(graph.eres[graph.stmts[item].head_id].label), set(graph.eres[graph.stmts[stmt_1].head_id].label))])

                            if len(same_head_label) == 1:
                                self_pair_entity_names.add(ere_id)
                                break
                            else:
                                same_head_label = set()
                                same_head_label.update([item for item in event_stmts if graph.eres[graph.stmts[item].head_id].category == 'Event' and set.intersection(set(graph.eres[graph.stmts[item].head_id].label), set(graph.eres[graph.stmts[stmt_2].head_id].label))])

                                if len(same_head_label) == 1:
                                    self_pair_entity_names.add(ere_id)
                                    break
                    elif {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Event', 'Relation'}:
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
                            same_head_label = set()
                            same_head_label.update([item for item in event_stmts if graph.eres[graph.stmts[item].head_id].category == 'Event' and set.intersection(set(graph.eres[graph.stmts[item].head_id].label), set(graph.eres[graph.stmts[stmt_1].head_id].label))])

                            if len(same_head_label) == 1:
                                self_pair_entity_names.add(ere_id)
                                break
                    # elif {graph.eres[graph.stmts[stmt_id].head_id].category for stmt_id in [stmt_1, stmt_2]} == {'Relation'}:
                    #     rel_ere_1 = graph.eres[graph.stmts[stmt_1].head_id]
                    #     rel_ere_2 = graph.eres[graph.stmts[stmt_2].head_id]
                    #     other_rel_neigh_1 = list(rel_ere_1.neighbor_ere_ids - {ere_id})[0]
                    #     other_rel_neigh_2 = list(rel_ere_2.neighbor_ere_ids - {ere_id})[0]
                    #     rel_ev_neighs_1 = {item for item in graph.eres[other_rel_neigh_1].neighbor_ere_ids if graph.eres[item].category == 'Event'}
                    #     rel_rel_neighs_1 = set.union(*[(graph.eres[item].neighbor_ere_ids - {other_rel_neigh_1}) for item in graph.eres[other_rel_neigh_1].neighbor_ere_ids if graph.eres[item].category == 'Relation']) - {other_rel_neigh_2}
                    #     rel_ev_neighs_2 = {item for item in graph.eres[other_rel_neigh_2].neighbor_ere_ids if graph.eres[item].category == 'Event'}
                    #     rel_rel_neighs_2 = set.union(*[(graph.eres[item].neighbor_ere_ids - {other_rel_neigh_2}) for item in graph.eres[other_rel_neigh_2].neighbor_ere_ids if graph.eres[item].category == 'Relation']) - {other_rel_neigh_1}
                    #
                    #     inter_eres = set.intersection(set.union(rel_ev_neighs_1, rel_rel_neighs_1), set.union(rel_ev_neighs_2, rel_rel_neighs_2))
                    #
                    #     rel_neighs_1 = set.union(rel_ev_neighs_1, rel_rel_neighs_1) - inter_eres
                    #     rel_neighs_2 = set.union(rel_ev_neighs_2, rel_rel_neighs_2) - inter_eres
                    #
                    #     if len(inter_eres - {ere_id}) >= max(0, (1 - len(rel_neighs_1))) + max(0, (1 - len(rel_neighs_2))):
                    #         same_head_label = set()
                    #         same_head_label.update([item for item in event_stmts if graph.eres[graph.stmts[item].head_id].category == 'Relation' and set.intersection(set(graph.eres[list(graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ere_id})[0]].label),
                    #                                                                                                                                                   set(graph.eres[list(graph.eres[graph.stmts[stmt_1].head_id].neighbor_ere_ids - {ere_id})[0]].label))])
                    #         if len(same_head_label) == 1:
                    #             self_pair_entity_names.add(ere_id)
                    #             break
                    #         else:
                    #             same_head_label = set()
                    #             same_head_label.update([item for item in event_stmts if graph.eres[graph.stmts[item].head_id].category == 'Relation' and set.intersection(set(graph.eres[list(graph.eres[graph.stmts[item].head_id].neighbor_ere_ids - {ere_id})[0]].label),
                    #                                                                                                                                                       set(graph.eres[list(graph.eres[graph.stmts[stmt_2].head_id].neighbor_ere_ids - {ere_id})[0]].label))])
                    #             if len(same_head_label) == 1:
                    #                 self_pair_entity_names.add(ere_id)
                    #                 break

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
def make_mixture_data(first_js_folder, second_js_folder, event_names, entity_names, event_type_maps, entity_type_maps, one_step_connectedness_map, two_step_connectedness_map,
                      out_data_dir, num_sources, num_query_hops, query_decay, num_shared_eres, data_size, size_cut, print_every, min_connectedness_one_step, min_connectedness_two_step, perc_train, perc_val, perc_test):
    verify_dir(out_data_dir)
    verify_dir(os.path.join(out_data_dir, 'Train'))
    verify_dir(os.path.join(out_data_dir, 'Val'))
    verify_dir(os.path.join(out_data_dir, 'Test'))

    counter = 0

    first_graph_list = load_all_graphs_in_folder(first_js_folder)

    if second_js_folder is not None:
        second_graph_list = load_all_graphs_in_folder(second_js_folder)
    else:
        second_graph_list = None

    num_train_graphs = math.ceil(perc_train * len(first_graph_list))
    num_test_graphs = math.floor(perc_val * len(first_graph_list))

    unused_graphs = set(first_graph_list.keys())

    train_graph_list = set(random.sample(list(unused_graphs), num_train_graphs))
    unused_graphs -= train_graph_list
    test_graph_list = set(random.sample(list(unused_graphs), num_test_graphs))
    unused_graphs -= test_graph_list
    val_graph_list = unused_graphs

    assert not set.intersection(train_graph_list, val_graph_list, test_graph_list)
    print(len(train_graph_list), len(val_graph_list), len(test_graph_list))

    train_entity_names = deepcopy(entity_names)
    val_entity_names = deepcopy(entity_names)
    test_entity_names = deepcopy(entity_names)

    filter_ent_names(train_entity_names, train_graph_list)
    filter_ent_names(val_entity_names, val_graph_list)
    filter_ent_names(test_entity_names, test_graph_list)

    train_self_pair_entity_names = process_ent_names(train_entity_names, first_graph_list, one_step_connectedness_map, two_step_connectedness_map)
    val_self_pair_entity_names = process_ent_names(val_entity_names, first_graph_list, one_step_connectedness_map, two_step_connectedness_map)
    test_self_pair_entity_names = process_ent_names(test_entity_names, first_graph_list, one_step_connectedness_map, two_step_connectedness_map)

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
                graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt = create_mix(first_graph_list, second_graph_list, event_names, entity_names, self_pair_entity_names, event_type_maps, entity_type_maps, one_step_connectedness_map, two_step_connectedness_map,
                                        num_sources, num_query_hops, query_decay, min_connectedness_one_step, min_connectedness_two_step, used_pairs, pos, tried_names, tried_ere_ids)

            self_pair_entity_names.discard(ent_ere_id)

            if counter < train_cut:
                dill.dump((graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt), open(os.path.join(out_data_dir, 'Train', re.sub('/', '_', re.sub('//', '_', ent_ere_id + '-self.p'))), 'wb'))
            elif counter < val_cut:
                dill.dump((graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt), open(os.path.join(out_data_dir, 'Val', re.sub('/', '_', re.sub('//', '_', ent_ere_id + '-self.p'))), 'wb'))
            else:
                dill.dump((graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt), open(os.path.join(out_data_dir, 'Test', re.sub('/', '_', re.sub('//', '_', ent_ere_id + '-self.p'))), 'wb'))
        else:
            source_graph, query_stmts, source_ere_id, ent_ere_ids, target_event_stmt = create_mix(first_graph_list, second_graph_list, event_names, entity_names, self_pair_entity_names, event_type_maps, entity_type_maps, one_step_connectedness_map,
                                    two_step_connectedness_map, num_sources, num_query_hops, query_decay, min_connectedness_one_step, min_connectedness_two_step, used_pairs, pos, tried_names, tried_ere_ids)
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
    parser.add_argument("--first_js_folder", type=str, default="/home/atomko/backup_drive/Summer_2020/Graph_Singles", help='First directory containing json graphs')
    parser.add_argument("--second_js_folder", type=str, default=None, help='Second directory contaning json graphs')
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

    make_mixture_data(first_js_folder, second_js_folder, event_names, entity_names, event_types, entity_types, one_step_connectedness_map, two_step_connectedness_map,
                      out_data_dir, num_sources, num_query_hops, query_decay, num_shared_eres, data_size, size_cut, print_every, min_connectedness_one_step, min_connectedness_two_step, perc_train, perc_val, perc_test)

    print("Done!\n")