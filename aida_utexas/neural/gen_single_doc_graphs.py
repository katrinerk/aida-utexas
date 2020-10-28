# Original author: Su Wang, 2019
# Modified by Alex Tomkovich in 2019/2020

######
# This script converts a folder of single-doc Wiki json KGs into pickled objects.
######

import dill
import argparse
import os
import json
import re

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

    # Print size info
    def __repr__(self):
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

# Ere class (represents EREs)
class Ere:
    def __init__(self, graph_id, category, ere_id, label):
        # <graph_id> describes the ID belonging to the single-doc source KG the ERE came from
        self.graph_id = graph_id
        # <category> is Event, Relation, or Entity
        self.category = category
        self.id = ere_id
        # <label> is a list of the names describing an ERE (e.g., 'United States', 'U.S.')
        self.label = label
        # <neighbor_ere_ids> describes the set of EREs which neighbor the ERE
        self.neighbor_ere_ids = set()
        # <stmt_ids> describes the set of stmts which are attached to the ERE
        self.stmt_ids = set()

    def __repr__(self):
        return "[ERE] | ID: %s | Label: %s" % (self.id, self.label)

    @staticmethod
    def entry_type():
        return "Ere"

# Stmt class (represents statements)
class Stmt:
    def __init__(self, graph_id, stmt_id, raw_label, label, head_id, tail_id):
        # <graph_id> describes the ID belonging to the single-doc source KG the stmt came from
        self.graph_id = graph_id
        self.id = stmt_id
        # <dup_ids> to hold statements which are identical in label/name/connections
        self.dup_ids = set()
        # <raw_label> is the unprocessed statement ontology label (e.g., "Movement.TransportPerson_Destination")
        self.raw_label = raw_label
        # <label> is a list containing the elements of the statement ontology label, split by . and _, e.g. ['Movement', 'TransportPerson', 'Destination']
        self.label = label
        # <head_id> describes the subject ERE node attached to the stmt
        self.head_id = head_id
        # <tail_id> describes the object ERE node attached to the stmt (None, if a typing stmt)
        self.tail_id = tail_id

    def __repr__(self):
        return "[STMT] | ID: %s | Label: %s" % (self.id, self.label)

    def link_info(self):
        return "Head: %s \nTail: %s" % (self.head_id, self.tail_id)

    @staticmethod
    def entry_type():
        return "Stmt"

# Graph class (composed of Ere and Stmt objects)
class Graph:
    def __init__(self, graph_id):
        self.graph_id = graph_id
        # A dictionary of Ere objects
        self.eres = dict()
        # A dictionary of Stmt objects
        self.stmts = dict()
        # The number of neighbor EREs reachable within 1 stmt traversal from each ERE
        self.connectedness_one_step = None
        # the number of neighbor EREs reachable within 2 stmt traversals from each ERE
        self.connectedness_two_step = None

    def __repr__(self):
        return "[GRAPH] | ID: %s | #N: %d | #E: %d" % (self.graph_id, len(self.eres), len(self.stmts))

    def is_empty(self):
        return len(self.eres) == 0 and len(self.stmts) == 0

    # Prepends graph ID to each ERE/stmt ID (so as to ensure EREs/stmts are distinct for different graphs)
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

            # Check if this statement already exists as another statement ID in the graph
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
    # Construct Graph object
    graph = Graph(graph_id)

    load_eres(graph, graph_js)
    load_statements(graph, graph_js)

    # Prune nodes with no neighbors from the graph
    remove_singletons(graph)

    # Compute one-step and two-step connectedness for all EREs in the graph
    compute_connectedness(graph)

    return graph

# Make a dir (if it doesn't already exist)
def verify_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, default="/home/cc/js-wiki-new-ontology", help='Input folder (abs path) containing single-doc Wiki json files')
    parser.add_argument("--out_dir", type=str, default="/home/cc/test_file_gen", help='Output folder (abs path) to contain pickled single-doc Wiki KGs')

    args = parser.parse_args()

    file_list = os.listdir(args.json_dir)

    verify_dir(args.out_dir)

    count = 0

    for file_name in file_list:
        file_path = os.path.join(args.json_dir, file_name)

        json_obj = json.load(open(file_path, 'r'))['theGraph']

        graph_id = file_name.split('.')[0]

        graph = read_graph(graph_id, json_obj)

        dill.dump(graph, open(os.path.join(args.out_dir, graph_id + '.p'), 'wb'))

        count += 1

        if count % 1000 == 0:
            print(str(count) + ' of ' + str(len(file_list)) + ' files processed...')

    print('Done processing single-doc KGs')