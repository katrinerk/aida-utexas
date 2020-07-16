""" Json, JsonJust and original text (i.e. rsd.txt files) to model inputs (i.e. document mixtures).

Author: Su Wang; 2019.
Usage:
python3 data_to_input.py \
    --<OPTION1>=... \
    --<OPTION2>=... \
    ...

"""

import itertools
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
    #seen_stmts = dict()

    for entry_id, entry in graph_js.items():
        if entry["type"] != "Statement":
            continue

        stmt_id = graph.unique_id(entry_id)
        subj_id = graph.unique_id(entry['subject'])

        if entry["predicate"] == "type":  # typing statements
            type_id = entry["object"]
            type_label = entry["object"].split("#")[-1]

            # tup = ('Type', subj_id, type_id)
            #
            # if tup in seen_stmts.keys():
            #     graph.stmts[seen_stmts[tup]].dup_ids.add(stmt_id)
            #     continue
            #
            # seen_stmts[tup] = stmt_id

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

            # tup = (' '.join(split_label), subj_id, obj_id)
            #
            # if tup in seen_stmts.keys():
            #     graph.stmts[seen_stmts[tup]].dup_ids.add(stmt_id)
            #     continue
            #
            # seen_stmts[tup] = stmt_id

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
def mix_graphs(graphs, num_shared_eres, min_connectedness_one_step, min_connectedness_two_step):
    count = 0

    # Retrieve a list of ere_ids which share type labels (and a corresponding "random_graph_index" indicating which graph is the target graph)
    shared_ere_type_ids_list = select_shared_ere_ids(graphs, num_shared_eres, min_connectedness_one_step, min_connectedness_two_step)

    if shared_ere_type_ids_list is None:
        return None

    shared_ere_type_ids_list, random_graph_index = shared_ere_type_ids_list
    target_graph = graphs[random_graph_index]
    target_graph_id = target_graph.graph_id

    query_index = 0  # Always use the first mixture point as the query, since it has the highest one-connectedness score sum

    mix_point_ere_ids = set()

    for shared_ere_type, shared_ere_ids in shared_ere_type_ids_list:
        target_ere_id = shared_ere_ids[random_graph_index]
        target_ere = target_graph.eres[target_ere_id]

        if count == query_index:
            query = retrieve_related_stmt_ids(target_graph, target_ere_id, 2)

        mix_point_ere_ids.add(target_ere_id)

        for i in range(len(graphs)):
            if i != random_graph_index:
                replace_ere(graphs[i], shared_ere_ids[i], target_ere)
        count += 1

    graph_mix = Graph("Mix")

    for ere_id in mix_point_ere_ids:
        graph_mix.eres[ere_id] = deepcopy(target_graph.eres[ere_id])

    for graph in graphs:
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

    for ere_id in mix_point_ere_ids:
        type_stmts = defaultdict(set)
        for (stmt_id, type_label) in [(stmt_id, (' ').join(graph_mix.stmts[stmt_id].label)) for stmt_id in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[stmt_id].tail_id is None]:
            type_stmts[type_label].add(stmt_id)

        for key in type_stmts.keys():
            if len(type_stmts[key]) > 1:
                if target_graph_id in [graph_mix.stmts[stmt_id].graph_id for stmt_id in type_stmts[key]]:
                    for stmt_id in type_stmts[key]:
                        if graph_mix.stmts[stmt_id].graph_id != target_graph_id:
                            graph_mix.eres[ere_id].stmt_ids.remove(stmt_id)
                            del graph_mix.stmts[stmt_id]
                else:
                    for stmt_id in list(type_stmts[key])[1:]:
                        graph_mix.eres[ere_id].stmt_ids.remove(stmt_id)
                        del graph_mix.stmts[stmt_id]

    origin_id = shared_ere_type_ids_list[query_index][1][random_graph_index]

    return origin_id, query, graph_mix, target_graph_id


# This function pre-loads all json graphs in the given folder
def load_all_graphs_in_folder(graph_js_folder):
    print('Loading all graphs in {}...'.format(graph_js_folder))

    graph_js_file_list = sorted([f for f in Path(graph_js_folder).iterdir() if f.is_file()])
    print('Found {} json files.'.format(len(graph_js_file_list)))

    graph_list = []

    for graph_js_file in tqdm(graph_js_file_list):
        if graph_js_file.is_file():
            graph_id = graph_js_file.name.split('.')[0]

            with open(graph_js_file, 'r') as fin:
                graph_js = json.load(fin)['theGraph']

            graph_list.append(read_graph(graph_id, graph_js))

    return graph_list


## This function generates graph mixtures from single-doc graphs chosen from the json folders it receives as parameters
#- If only first_js_folder is specified, make_mixture_data() will generate self-mixtures (i.e., it will mix only single-docs from first_js_folder)
#- If second_js_folder is also specified, make_mixture_data() will generate mixtures from single-docs taken from both first_js_folder and second_js_folder
def make_mixture_data(first_js_folder, second_js_folder, out_data_dir, num_sources, num_shared_eres, data_size, size_cut, print_every, min_connectedness_one_step, min_connectedness_two_step, perc_train, perc_val, perc_test):
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

    start = time.time()

    used_pairs = set()

    train_cut = perc_train * data_size
    val_cut = train_cut + (perc_val * data_size)

    used_singles = [set()] * 2

    while counter < data_size:
        if counter == train_cut:
            first_graph_list = [graph for graph in first_graph_list if graph.graph_id not in used_singles[0]]
            if second_graph_list:
                second_graph_list = [graph for graph in second_graph_list if graph.graph_id not in used_singles[0]]
        elif counter == val_cut:
            first_graph_list = [graph for graph in first_graph_list if graph.graph_id not in used_singles[1]]
            if second_graph_list:
                second_graph_list = [graph for graph in second_graph_list if graph.graph_id not in used_singles[1]]

        graphs = load_random_entries(first_graph_list, second_graph_list, num_sources, used_pairs)

        mix_result = mix_graphs(graphs, num_shared_eres, min_connectedness_one_step, min_connectedness_two_step)
        used_pairs.add(frozenset([graph.graph_id for graph in graphs]))

        if mix_result is None:
            continue
        target_ere_id, query, graph_mix, target_graph_id = mix_result

        # Reject mixtures of size (ERE's + statements) > size_cut
        if len(query) == 0 or (len(graph_mix.eres.keys()) + len(graph_mix.stmts.keys()) > size_cut):
            continue

        # Reject mixtures containing no additional target graph statements to be extracted (i.e., the only target graph statements in the mixture are those found in the query)
        if len(set([stmt_id for stmt_id in graph_mix.stmts.keys() if graph_mix.stmts[stmt_id].graph_id == target_graph_id]) - set(query)) == 0:
            continue

        if counter < train_cut:
            used_singles[0].update([graph.graph_id for graph in graphs])
        elif counter < val_cut:
            used_singles[1].update([graph.graph_id for graph in graphs])

        file_name = '-'.join(graph.graph_id for graph in graphs)

        if counter < train_cut:
            dill.dump((target_ere_id, query, graph_mix, target_graph_id), open(os.path.join(out_data_dir, "Train", file_name) + ".p", "wb"))
        elif counter < val_cut:
            dill.dump((target_ere_id, query, graph_mix, target_graph_id), open(os.path.join(out_data_dir, "Val", file_name) + ".p", "wb"))
        else:
            dill.dump((target_ere_id, query, graph_mix, target_graph_id), open(os.path.join(out_data_dir, "Test", file_name) + ".p", "wb"))

        counter += 1
        if counter % print_every == 0:
            print("... processed %d entries (%.2fs)." % (counter, time.time() - start))
            start = time.time()

    dill.dump(used_singles, open(os.path.join(out_data_dir, "used_singles.p"), "wb"))

    print("\nDone!\n")
