# Created by Su Wang, 2019
# Modified by Alex Tomkovich

######
# This file generates graph salads (artificial mixtures of source KGs which are merged at common
# entities/events.
######

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from copy import deepcopy
from operator import itemgetter
from pathlib import Path

import dill
from tqdm import tqdm
import itertools
import math

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

# Load the EREs from a graph JSON
def load_eres(graph, graph_js):
    for entry_id, entry in graph_js.items():
        # Ignore all Statements, SameAsClusters, and ClusterMemberships first
        if entry["type"] in ["Statement", "SameAsCluster", "ClusterMembership"]:
            continue

        ere_id = graph.unique_id(entry_id)

        # Process relation nodes
        if entry["type"] == "Relation":
            graph.eres[ere_id] = Ere(
                graph_id=graph.graph_id,
                category=entry["type"],
                ere_id=ere_id,
                label=["Relation"] # Relation nodes have no explicit name
            )
        # Process event/entity nodes
        else:
            keep_labels = []

            if "name" in entry.keys():
                for label in entry["name"]:
                    match = re.search('[_ ]+', label)

                    # Ensure that the name is not composed entirely of underscores and/or whitespace
                    if match and match.span()[1] - match.span()[0] == len(label):
                        continue

                    keep_labels.append(label)

            graph.eres[ere_id] = Ere(
                graph_id=graph.graph_id,
                category=entry["type"],
                ere_id=ere_id,
                label=keep_labels
            )

# Load the statements from a graph JSON
def load_statements(graph, graph_js):
    seen_stmts = dict()

    for entry_id, entry in graph_js.items():
        if entry["type"] != "Statement":
            continue

        stmt_id = graph.unique_id(entry_id)
        subj_id = graph.unique_id(entry['subject'])

        # Process typing statements
        if entry["predicate"] == "type":
            type_id = entry["object"]
            type_label = entry["object"].split("#")[-1]

            tup = ('Type', subj_id, type_id)

            # Check if this statement already exists as another statement ID
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
        # Processing non-typing (event or relation) statements
        else:
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

# Remove singleton nodes (nodes with no neighbors)
def remove_singletons(graph):
    singleton_ids = [ere_id for ere_id in graph.eres.keys() if len(graph.eres[ere_id].neighbor_ere_ids) == 0]

    for singleton_id in singleton_ids:
        for stmt_id in graph.eres[singleton_id].stmt_ids:
            del graph.stmts[stmt_id]
        del graph.eres[singleton_id]

# Compute one- and two-step connectedness scores for all EREs in a graph
def compute_connectedness(graph):
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

# Read in a graph (from a JSON)
def read_graph(graph_id, graph_js):
    graph = Graph(graph_id)

    load_eres(graph, graph_js)
    load_statements(graph, graph_js)

    remove_singletons(graph)

    compute_connectedness(graph)

    return graph

# Make a dir (if it doesn't already exist)
def verify_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

##### GRAPH SALAD CREATION #####

# Create an initial query set around a given ERE
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

# Checks whether <new_ere_id> is reachable from <target_ere_id> in <graph>
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

# Replace the node with ID <source_id> with <target_ere> in <graph>
def replace_ere(graph, source_id, target_ere):
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

def abridge_graph(origin_id, query, graph_mix, target_graph_id, num_abridge_hops):
    for ere_id in graph_mix.eres.keys():
        graph_mix.eres[ere_id].stmt_ids.update({item for item in graph_mix.stmts.keys() if ere_id in {graph_mix.stmts[item].head_id, graph_mix.stmts[item].tail_id}})
        graph_mix.eres[ere_id].neighbor_ere_ids = set.union(*[{graph_mix.stmts[item].head_id, graph_mix.stmts[item].tail_id} for item in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[item].tail_id]) - {ere_id}

    for ere_id, ere in graph_mix.eres.items():
        items_to_del = set()

        if len([item for item in ere.stmt_ids if not graph_mix.stmts[item].tail_id]) > 1:
            for item in ere.stmt_ids:
                if not graph_mix.stmts[item].tail_id and graph_mix.stmts[item].graph_id != target_graph_id:
                    items_to_del.add(item)

        for item in items_to_del:
            del graph_mix.stmts[item]

            graph_mix.eres[ere_id].stmt_ids.discard(item)

    merge_points = {item for item in graph_mix.eres.keys() if len({graph_mix.stmts[stmt_id].graph_id for stmt_id in graph_mix.eres[item].stmt_ids}) == 3}

    seen_eres = set()
    curr_eres = deepcopy(merge_points)

    i = 0

    while len(curr_eres) > 0 and i < num_abridge_hops:
        seen_eres.update(curr_eres)
        curr_eres = set.union(*[graph_mix.eres[ere_id].neighbor_ere_ids for ere_id in curr_eres]) - seen_eres

        i += 1

    reachable_stmts = set.union(*[graph_mix.eres[ere_id].stmt_ids for ere_id in seen_eres])
    seen_eres.update(curr_eres)

    for ere_id in deepcopy(seen_eres):
        if graph_mix.eres[ere_id].category == 'Relation':
            reachable_stmts.update(graph_mix.eres[ere_id].stmt_ids)
            seen_eres.update(graph_mix.eres[ere_id].neighbor_ere_ids)

    reachable_stmts.update(set.union(*[{item for item in graph_mix.eres[ere_id].stmt_ids if not graph_mix.stmts[item].tail_id} for ere_id in seen_eres]))

    for ere_id in set(graph_mix.eres.keys()) - seen_eres:
        del graph_mix.eres[ere_id]

    for stmt_id in set(graph_mix.stmts.keys()) - reachable_stmts:
        del graph_mix.stmts[stmt_id]

    for ere_id in graph_mix.eres.keys():
        graph_mix.eres[ere_id].stmt_ids = set.intersection(graph_mix.eres[ere_id].stmt_ids, reachable_stmts)
        graph_mix.eres[ere_id].neighbor_ere_ids = set.union(*[{graph_mix.stmts[item].head_id, graph_mix.stmts[item].tail_id} for item in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[item].tail_id]) - {ere_id}

    for stmt_id, stmt in graph_mix.stmts.items():
        head_id = stmt.head_id
        tail_id = stmt.tail_id

        if not tail_id:
            assert stmt_id in graph_mix.eres[head_id].stmt_ids
        else:
            assert stmt_id in graph_mix.eres[head_id].stmt_ids
            assert stmt_id in graph_mix.eres[tail_id].stmt_ids
            assert tail_id in graph_mix.eres[head_id].neighbor_ere_ids
            assert head_id in graph_mix.eres[tail_id].neighbor_ere_ids

    for ere_id, ere in graph_mix.eres.items():
        assert ere_id not in ere.neighbor_ere_ids

        assert len([item for item in ere.stmt_ids if not graph_mix.stmts[item].tail_id]) == 1
        assert graph_mix.stmts[[item for item in ere.stmt_ids if not graph_mix.stmts[item].tail_id][0]].graph_id == ere.graph_id

        assert set.union(*[{graph_mix.stmts[stmt_id].head_id, graph_mix.stmts[stmt_id].tail_id} if graph_mix.stmts[stmt_id].tail_id else {graph_mix.stmts[stmt_id].head_id} for stmt_id in ere.stmt_ids]) - {ere_id} == ere.neighbor_ere_ids

        temp = set()

        for stmt_id, stmt in graph_mix.stmts.items():
            if ere_id in [stmt.head_id, stmt.tail_id]:
                temp.add(stmt_id)

        assert temp == ere.stmt_ids

def get_query_points(sample_graphs, name_dict, src_to_name_map, num_sources, two_step_connectedness_map, max_connectedness_two_step, ere_type_maps):
    # Determine all ERE names which the chosen source graphs share
    shared_names = set.intersection(*[src_to_name_map[item] for item in sample_graphs])
    name_to_ere_dict = defaultdict(lambda: defaultdict(set))

    # Create a dictionary mapping <source graph> and <name> keys to a list of (ere_id, two-step connectedness) tuples
    for name in shared_names:
        for ere_id in name_dict[name]:
            if ere_id.split('_h')[0] in sample_graphs:
                name_to_ere_dict[ere_id.split('_h')[0]][name].add((ere_id, two_step_connectedness_map[ere_id]))

    merge_comb_dict = dict()
    ranked_dict = dict()
    super_ranked_list = []

    for name in shared_names:
        # Determine all possible combinations of mergeable EREs with a particular name for the set of chosen source graphs
        merge_comb_dict[name] = set(itertools.product(*[name_to_ere_dict[graph_id][name] for graph_id in sample_graphs]))

        # Ensure the total connectedness of all merge candidates does not exceed the specified maximum (if any)
        if max_connectedness_two_step:
            merge_comb_dict[name] = {item for item in merge_comb_dict[name] if sum([sub_item[1] for sub_item in item]) <= max_connectedness_two_step}

        if len(merge_comb_dict[name]) > 0:
            # Rank each combination of mergeable candidates by their total two-step connectedness for the current name
            ranked_dict[name] = sorted(list(merge_comb_dict[name]), key=lambda x: sum([item[1] for item in x]), reverse=True)
            # Add the most highly connected (two-step) combination for the current name to a master list
            super_ranked_list.append((name, ranked_dict[name][0]))

    # Globally rank each selected combination of merge candidates across all shared event names
    super_ranked_list = sorted(super_ranked_list, key=lambda x: sum([item[1] for item in x[1]]), reverse=True)

    query_points = []
    seen_eres = set()

    for item in super_ranked_list:
        # A given ERE should be used to create no more than one merge point
        if len(set([sub_item[0] for sub_item in item[1]]) - seen_eres) != num_sources:
            continue
        # All EREs in the combination should share a type
        elif not set.intersection(*[ere_type_maps[ere_id] for ere_id in [sub_item[0] for sub_item in item[1]]]):
            continue
        else:
            query_points.append(item)
            seen_eres.update(set([sub_item[0] for sub_item in item[1]]))

    return query_points

def get_possible_target_graph_ids(graph_list, sample_graphs, query_points):
    poss_target_graph_ids = []

    for graph_iter, graph_id in enumerate(sample_graphs):
        root_ere_id = query_points[0][1][graph_iter][0]

        reach = True

        count = query_points[0][1][graph_iter][1]

        for name_iter in range(1, len(query_points)):
            if not reachable(graph_list[graph_id], root_ere_id, query_points[name_iter][1][graph_iter][0]):
                reach = False
                break
            else:
                count += query_points[name_iter][1][graph_iter][1]

        if reach:
            poss_target_graph_ids.append((graph_iter, graph_id, count))

    return poss_target_graph_ids

# This function creates a graph salad by mixing a set of single-doc graphs at <num_shared_eres> or more points.
def create_mix(graph_list, event_names, entity_names, event_type_maps, entity_type_maps, one_step_connectedness_map, two_step_connectedness_map, max_connectedness_two_step, event_src_to_name_map, entity_src_to_name_map,
                                           event_name_counts, entity_name_counts, num_sources, num_shared_eres, num_abridge_hops, used_pairs):
    # Determine all event names which are found in <num_sources> or more source docs
    mixable_events = dict({key: value for (key, value) in event_name_counts if value >= num_sources})

    done = False

    while not done:
        # Sample a random event name from those which are found in <num_sources> or more source graphs
        event_name = random.sample(list(mixable_events.keys()), 1)[0]
        poss_eres = event_names[event_name]

        # Sample three ERE IDs from the set associated with the chosen event name;
        # the three sources from which these EREs come will be the source graphs used to create the salad
        sample_eres = random.sample(poss_eres, num_sources)
        sample_graphs = [item.split('_h')[0] for item in sample_eres]

        # Ensure that (i) all source graphs from which the chosen EREs are drawn are in the allowed (train, val, test) subset of sources and
        #             (ii) the three EREs chosen are from three distinct source graphs
        if (set(sample_graphs) - set(graph_list.keys())) or (len(set(sample_graphs)) < num_sources):
            continue

        query_event_points = get_query_points(sample_graphs, event_names, event_src_to_name_map, num_sources, two_step_connectedness_map, max_connectedness_two_step, event_type_maps)
        query_entity_points = get_query_points(sample_graphs, entity_names, entity_src_to_name_map, num_sources, two_step_connectedness_map, max_connectedness_two_step, entity_type_maps)

        # Use the <num_shared_eres> query points with the highest two-step connectedness scores
        query_event_points = query_event_points[:num_shared_eres]

        if len(query_event_points) < num_shared_eres:
            continue

        # Determine which source graphs can serve as target graphs in separate instances (given the reachability constraint)
        poss_target_graph_ids = get_possible_target_graph_ids(graph_list, sample_graphs, query_event_points)

        if len(poss_target_graph_ids) == 0:
            continue

        # Ensure that at least one of these salad instances has not previously been created
        target_graph_ids = [item[1] for item in poss_target_graph_ids]
        source_graph_ids_w_targets = {(frozenset(sample_graphs), item) for item in target_graph_ids}

        if source_graph_ids_w_targets - used_pairs:
            done = True

            # Filter out all salad-target instances that have been previously created
            new_salad_targets = [item[1] for item in source_graph_ids_w_targets - used_pairs]
            poss_target_graph_ids = [item for item in poss_target_graph_ids if item[1] in new_salad_targets]

    # Find all potential entity merge points which are reachable from the event merge points
    reachable_ents = defaultdict(set)

    for graph_iter, graph_id in [(item[0], item[1]) for item in poss_target_graph_ids]:
        root_ere_id = query_event_points[0][1][graph_iter][0]

        for name_iter in range(len(query_entity_points)):
            if reachable(graph_list[graph_id], root_ere_id, query_entity_points[name_iter][1][graph_iter][0]):
                reachable_ents[graph_id].add(name_iter)

    # Mix the source graphs and create the salads
    graph_info = []

    for (target_graph_iter, target_graph_id, count) in poss_target_graph_ids:
        graph_copies = [deepcopy(graph_list[item]) for item in sample_graphs]

        # Randomly choose one of the event merge points (the "origin ID") to construct the query set
        random_start_ind = random.sample(range(len(query_event_points)), 1)[0]

        for point_iter, (name, ere_info) in enumerate(query_event_points):
            target_ere_id = ere_info[target_graph_iter][0]
            target_ere = graph_copies[target_graph_iter].eres[target_ere_id]

            # If the current merge point is the origin ID, construct the query set
            if point_iter == random_start_ind:
                query = retrieve_related_stmt_ids(graph_copies[target_graph_iter], target_ere_id, 2)

            # Replace the merge ERE in each source graph with the target graph's ERE
            for (graph_iter, other_graph) in [(iter, item) for (iter, item) in enumerate(graph_copies) if iter != target_graph_iter]:
                source_ere_id = ere_info[graph_iter][0]
                replace_ere(graph_copies[graph_iter], source_ere_id, target_ere)

        # Merge all entity EREs which are reachable from the event EREs
        for name_iter in reachable_ents[target_graph_id]:
            ere_info = query_entity_points[name_iter][1]
            target_ere_id = ere_info[target_graph_iter][0]
            target_ere = graph_copies[target_graph_iter].eres[target_ere_id]

            for (graph_iter, other_graph) in [(iter, item) for (iter, item) in enumerate(graph_copies) if iter != target_graph_iter]:
                source_ere_id = ere_info[graph_iter][0]
                replace_ere(graph_copies[graph_iter], source_ere_id, target_ere)

        graph_mix = Graph("Mix")

        # Fetch the ERE in the target graph for each merge point
        for ere_id in [item[1][target_graph_iter][0] for item in query_event_points]:
            graph_mix.eres[ere_id] = deepcopy(graph_copies[target_graph_iter].eres[ere_id])

        mix_point_ere_ids = set([item[1][target_graph_iter][0] for item in query_event_points])

        # Merge selected EREs and add subgraphs surrounding merge points from each component graph
        for graph in graph_copies:
            eres_to_update = set()
            stmts_to_update = set()

            init_stmts = set.union(*[set(graph.eres[mix_point_ere_id].stmt_ids) for mix_point_ere_id in mix_point_ere_ids])
            stmts_to_update.update(init_stmts)

            target_ids = set.union(*[set(graph.eres[mix_point_ere_id].neighbor_ere_ids) for mix_point_ere_id in mix_point_ere_ids])
            target_ids = list(target_ids - mix_point_ere_ids)
            seen_eres = mix_point_ere_ids.copy()

            while len(target_ids) > 0:
                new_target_set = set()
                for target_id in target_ids:
                    if target_id not in mix_point_ere_ids:
                        eres_to_update.add(target_id)
                    seen_eres.add(target_id)

                    stmts_to_update.update(graph.eres[target_id].stmt_ids)

                    new_target_set.update(graph.eres[target_id].neighbor_ere_ids)
                new_target_set -= seen_eres
                target_ids = list(new_target_set)

            graph_mix.eres.update({ere_id: graph.eres[ere_id] for ere_id in eres_to_update})
            graph_mix.stmts.update({stmt_id: graph.stmts[stmt_id] for stmt_id in stmts_to_update})

            for ere_id in mix_point_ere_ids:
                graph_mix.eres[ere_id].neighbor_ere_ids.update(graph.eres[ere_id].neighbor_ere_ids)
                graph_mix.eres[ere_id].stmt_ids.update(graph.eres[ere_id].stmt_ids)

        # Ensure that there are no duplicate type statements from different component graphs
        for ere_id in mix_point_ere_ids:
            type_stmts = defaultdict(set)
            for (stmt_id, type_label) in [(stmt_id, (' ').join(graph_mix.stmts[stmt_id].label)) for stmt_id in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[stmt_id].tail_id is None]:
                type_stmts[type_label].add(stmt_id)

            for key in type_stmts.keys():
                if len(type_stmts[key]) > 1:
                    if sample_graphs[target_graph_iter] in [graph_mix.stmts[stmt_id].graph_id for stmt_id in type_stmts[key]]:
                        for stmt_id in type_stmts[key]:
                            if graph_mix.stmts[stmt_id].graph_id != sample_graphs[target_graph_iter]:
                                graph_mix.eres[ere_id].stmt_ids.remove(stmt_id)
                                del graph_mix.stmts[stmt_id]
                    else:
                        for stmt_id in list(type_stmts[key])[1:]:
                            graph_mix.eres[ere_id].stmt_ids.remove(stmt_id)
                            del graph_mix.stmts[stmt_id]

        origin_id = query_event_points[random_start_ind][1][target_graph_iter][0]

        # If requested, reduce the size of the graph by cropping subgraphs pieces more than <num_abridge_hops> away from any given event merge point
        if num_abridge_hops:
            abridge_graph(origin_id, query, graph_mix, target_graph_id, num_abridge_hops)

        graph_info.append((origin_id, query, graph_mix, target_graph_id))

    return graph_info, sample_graphs

# This function pre-loads all json graphs in the given folder
def load_all_graphs_in_folder(graph_js_folder):
    print('Loading all graphs in {}...'.format(graph_js_folder))

    graph_file_list = sorted([f for f in Path(graph_js_folder).iterdir() if f.is_file()])

    graph_list = dict()

    for graph_file in tqdm(graph_file_list):
        if graph_file.is_file():
            graph_list[str(graph_file).split('.p')[0].split('/')[-1]] = dill.load(open(graph_file, 'rb'))

    return graph_list

# Filter out names based on subset of source graphs assigned to train, val, test partitions
def filter_names(names, graph_list):
    keys_to_del = []

    for key in names.keys():
        eres_to_dis = set()

        for ere_id in names[key]:
            if ere_id.split('_h')[0] not in graph_list:
                eres_to_dis.add(ere_id)

        names[key] -= eres_to_dis

        if len(names[key]) == 0:
            keys_to_del.append(key)

    for key in keys_to_del:
        del names[key]

# Sort through all event nodes, discarding those which do not fulfill the requirements to be candidates for merging.
# Event node candidates for merging must have:
# --At least one attached non-typing statement (by necessity, attached to an entity node)
# --The given minimum one-step and two-step connectedness scores
def filter_merge_candidates(ere_name_map, graph_list, one_step_connectedness_map, two_step_connectedness_map, min_connectedness_one_step, min_connectedness_two_step):
    keys_to_del = []

    for key in ere_name_map.keys():
        eres_to_dis = []

        for ere_id in ere_name_map[key]:
            graph = graph_list[ere_id.split('_h')[0]]

            event_stmt_ids = {stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Event')}

            if (len(event_stmt_ids) < 1) or (one_step_connectedness_map[ere_id] < min_connectedness_one_step) or (two_step_connectedness_map[ere_id] < min_connectedness_two_step):
                eres_to_dis.append(ere_id)

        for ere_id in eres_to_dis:
            ere_name_map[key].discard(ere_id)

        if len(ere_name_map[key]) == 0:
            keys_to_del.append(key)

    for key in keys_to_del:
        del ere_name_map[key]

# Organize (train, val, test) partitions and write graph salads to disk
def make_mixture_data(wiki_js_folder, event_name_map, entity_name_map, event_type_maps, entity_type_maps, one_step_connectedness_map, two_step_connectedness_map,
                      out_data_dir, num_sources, num_shared_eres, num_abridge_hops, data_size, max_size, print_every, min_connectedness_one_step, min_connectedness_two_step, max_connectedness_two_step, perc_train, perc_test):
    verify_dir(out_data_dir)
    verify_dir(os.path.join(out_data_dir, 'Train'))
    verify_dir(os.path.join(out_data_dir, 'Val'))
    verify_dir(os.path.join(out_data_dir, 'Test'))

    graph_list = load_all_graphs_in_folder(wiki_js_folder)

    # Filter out EREs which do not meet requirements for merging
    filter_merge_candidates(event_name_map, graph_list, one_step_connectedness_map, two_step_connectedness_map, min_connectedness_one_step, min_connectedness_two_step)
    filter_merge_candidates(entity_name_map, graph_list, one_step_connectedness_map, two_step_connectedness_map, min_connectedness_one_step, min_connectedness_two_step)

    num_train_graphs = math.ceil(perc_train * len(graph_list))
    num_test_graphs = math.ceil(perc_test * len(graph_list))

    unused_graphs = set(graph_list.keys())

    train_graph_list = set(random.sample(list(unused_graphs), num_train_graphs))
    unused_graphs -= train_graph_list
    test_graph_list = set(random.sample(list(unused_graphs), num_test_graphs))
    unused_graphs -= test_graph_list
    val_graph_list = unused_graphs

    assert not set.intersection(train_graph_list, val_graph_list, test_graph_list)
    print(len(train_graph_list), len(val_graph_list), len(test_graph_list))

    # Assign subsets of source graphs for use in creating each of the (train, val, test) partitions.
    train_event_name_map = deepcopy(event_name_map)
    val_event_name_map = deepcopy(event_name_map)
    test_event_name_map = deepcopy(event_name_map)

    filter_names(train_event_name_map, train_graph_list)
    filter_names(val_event_name_map, val_graph_list)
    filter_names(test_event_name_map, test_graph_list)

    train_entity_name_map = deepcopy(entity_name_map)
    val_entity_name_map = deepcopy(entity_name_map)
    test_entity_name_map = deepcopy(entity_name_map)

    filter_names(train_entity_name_map, train_graph_list)
    filter_names(val_entity_name_map, val_graph_list)
    filter_names(test_entity_name_map, test_graph_list)

    # Create lists of (ere_name, <num of sources containing ERE with ere_name>) tuples, sorted by num of sources
    train_event_name_counts = sorted([(item, len(set([ere_id.split('_h')[0] for ere_id in train_event_name_map[item]]))) for item in train_event_name_map.keys()], key=lambda x: x[1], reverse=True)
    train_entity_name_counts = sorted([(item, len(set([ere_id.split('_h')[0] for ere_id in train_entity_name_map[item]]))) for item in train_entity_name_map.keys()], key=lambda x: x[1], reverse=True)

    val_event_name_counts = sorted([(item, len(set([ere_id.split('_h')[0] for ere_id in val_event_name_map[item]]))) for item in val_event_name_map.keys()], key=lambda x: x[1], reverse=True)
    val_entity_name_counts = sorted([(item, len(set([ere_id.split('_h')[0] for ere_id in val_entity_name_map[item]]))) for item in val_entity_name_map.keys()], key=lambda x: x[1], reverse=True)

    test_event_name_counts = sorted([(item, len(set([ere_id.split('_h')[0] for ere_id in test_event_name_map[item]]))) for item in test_event_name_map.keys()], key=lambda x: x[1], reverse=True)
    test_entity_name_counts = sorted([(item, len(set([ere_id.split('_h')[0] for ere_id in test_entity_name_map[item]]))) for item in test_entity_name_map.keys()], key=lambda x: x[1], reverse=True)

    train_event_src_to_name_map = defaultdict(set)
    train_entity_src_to_name_map = defaultdict(set)
    val_event_src_to_name_map = defaultdict(set)
    val_entity_src_to_name_map = defaultdict(set)
    test_event_src_to_name_map = defaultdict(set)
    test_entity_src_to_name_map = defaultdict(set)

    for key, item in train_event_name_map.items():
        for ere_id in item:
            train_event_src_to_name_map[ere_id.split('_h')[0]].add(key)

    for key, item in train_entity_name_map.items():
        for ere_id in item:
            train_entity_src_to_name_map[ere_id.split('_h')[0]].add(key)

    for key, item in val_event_name_map.items():
        for ere_id in item:
            val_event_src_to_name_map[ere_id.split('_h')[0]].add(key)

    for key, item in val_entity_name_map.items():
        for ere_id in item:
            val_entity_src_to_name_map[ere_id.split('_h')[0]].add(key)

    for key, item in test_event_name_map.items():
        for ere_id in item:
            test_event_src_to_name_map[ere_id.split('_h')[0]].add(key)

    for key, item in test_entity_name_map.items():
        for ere_id in item:
            test_entity_src_to_name_map[ere_id.split('_h')[0]].add(key)

    random.seed(1)

    used_pairs = set()

    train_cut = perc_train * data_size
    val_cut = train_cut + ((1 - (perc_train + perc_test)) * data_size)

    start = time.time()

    event_name_map = train_event_name_map
    entity_name_map = train_entity_name_map
    event_name_counts = train_event_name_counts
    entity_name_counts = train_entity_name_counts
    event_src_to_name_map = train_event_src_to_name_map
    entity_src_to_name_map = train_entity_src_to_name_map

    counter = 0

    while counter < data_size:
        if counter == train_cut:
            event_name_map = val_event_name_map
            entity_name_map = val_entity_name_map
            event_name_counts = val_event_name_counts
            entity_name_counts = val_entity_name_counts
            event_src_to_name_map = val_event_src_to_name_map
            entity_src_to_name_map = val_entity_src_to_name_map
        elif counter == val_cut:
            event_name_map = test_event_name_map
            entity_name_map = test_entity_name_map
            event_name_counts = test_event_name_counts
            entity_name_counts = test_entity_name_counts
            event_src_to_name_map = test_event_src_to_name_map
            entity_src_to_name_map = test_entity_src_to_name_map

        graph_info, used_graph_ids = create_mix(graph_list, event_name_map, entity_name_map, event_type_maps, entity_type_maps, one_step_connectedness_map, two_step_connectedness_map, max_connectedness_two_step, event_src_to_name_map, entity_src_to_name_map,
                                           event_name_counts, entity_name_counts, num_sources, num_shared_eres, num_abridge_hops, used_pairs)

        # Used to ensure we don't create more than one salad with the same three source graphs
        used_pairs.update([(frozenset(used_graph_ids), item[3]) for item in graph_info])

        for item in graph_info:
            origin_id, query, graph_mix, target_graph_id = item

            # Reject salads with no query set
            if len(query) == 0:
                continue

            # Reject salads exceeding the maximum size (where max_size is in KB)
            if len(dill.dumps(graph_mix, -1)) >= (814.433 * max_size):
                continue

            # Reject mixtures containing no additional target graph statements to be extracted (i.e., the only target graph statements in the mixture are those found in the query)
            if len(set([stmt_id for stmt_id in graph_mix.stmts.keys() if graph_mix.stmts[stmt_id].graph_id == target_graph_id]) - set(query)) == 0:
                continue

            file_name = '-'.join(used_graph_ids) + '_target-' + target_graph_id

            if counter < train_cut:
                dill.dump((origin_id, query, graph_mix, target_graph_id), open(os.path.join(out_data_dir, "Train", file_name) + ".p", "wb"))
            elif counter < val_cut:
                dill.dump((origin_id, query, graph_mix, target_graph_id), open(os.path.join(out_data_dir, "Val", file_name) + ".p", "wb"))
            else:
                dill.dump((origin_id, query, graph_mix, target_graph_id), open(os.path.join(out_data_dir, "Test", file_name) + ".p", "wb"))

            counter += 1
            if counter % print_every == 0:
                print("... processed %d entries (%.2fs)." % (counter, time.time() - start))
                start = time.time()

    print("\nDone!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_data_dir", type=str, default="/home/atomko/backup_drive/Summer_2020/Test_Debug_Mix",
                        help='Folder where the mixtures will be written (will be created by the script, if it does not already exist)')
    parser.add_argument("--wiki_js_folder", type=str, default="/home/atomko/backup_drive/Summer_2020/Graph_Singles",
                        help='Directory containing Wiki json graphs')
    parser.add_argument("--event_name_maps", type=str, default="/home/atomko/backup_drive/Summer_2020/event_names_uncollapsed.p",
                        help='Name mapping for events')
    parser.add_argument("--entity_name_maps", type=str, default="/home/atomko/backup_drive/Summer_2020/entity_names_uncollapsed.p",
                        help='Name mapping for entities')
    parser.add_argument("--event_type_maps", type=str, default="/home/atomko/backup_drive/Summer_2020/event_types_uncollapsed.p",
                        help='Type mapping for events')
    parser.add_argument("--entity_type_maps", type=str, default="/home/atomko/backup_drive/Summer_2020/entity_types_uncollapsed.p",
                        help='Type mapping for entities')
    parser.add_argument("--one_step_connectedness_map", type=str, default="/home/atomko/backup_drive/Summer_2020/connectedness_one_step_uncollapsed.p",
                        help='One-step connectedness map')
    parser.add_argument("--two_step_connectedness_map", type=str, default="/home/atomko/backup_drive/Summer_2020/connectedness_two_step_uncollapsed.p",
                        help='Two-step connectedness map')
    parser.add_argument("--num_sources", type=int, default=3,
                        help='Number of single docs to mix at one time')
    parser.add_argument("--num_shared_eres", type=int, default=3,
                        help='Required number of mixture points in a produced mixture')
    parser.add_argument("--num_abridge_hops", type=int, default=1000,
                        help='Required number of mixture points in a produced mixture')
    parser.add_argument("--data_size", type=int, default=1000,
                        help='Number of mixtures to create')
    parser.add_argument("--max_size", type=int, default=1500,
                        help='Maximum size of graph mixtures (in KB)')
    parser.add_argument("--print_every", type=int, default=100,
                        help='Print every x mixtures created')
    parser.add_argument("--min_connectedness_one_step", type=int, default=2,
                        help='The minimum one-step connectedness score for an ERE to be selected as a mixture point')
    parser.add_argument("--min_connectedness_two_step", type=int, default=4,
                        help='The minimum two-step connectedness score for an ERE to be selected as a mixture point')
    parser.add_argument("--max_connectedness_two_step", type=int, default=60,
                        help='Maximum two-step connectedness of merge points')
    parser.add_argument("--perc_train", type=float, default=.8,
                        help='Percentage of <data_size> mixtures to assign to the training set')
    parser.add_argument("--perc_val", type=float, default=.1,
                        help='Percentage of <data_size> mixtures to assign to the validation set')
    parser.add_argument("--perc_test", type=float, default=.1,
                        help='Percentage of <data_size> mixtures to assign to the test set')

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
                      out_data_dir, num_sources, num_shared_eres, num_abridge_hops, data_size, max_size, print_every, min_connectedness_one_step, min_connectedness_two_step, max_connectedness_two_step, perc_train, perc_val, perc_test)

    print("Done!\n")
