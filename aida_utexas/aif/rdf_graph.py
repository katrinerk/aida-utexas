"""
Author: Katrin Erk, Oct 2018
- Takes as input an AIDA interchange format file, reads it via the rdflib library, and
transforms it into a mapping from node ids to node entries for each subject of the graph.

- Each node entry contains a list of predicate/object pairs for its outgoing edges, and a list of
predicate/subject pairs for its incoming edges.

- Here is how rdflib reads an AIDA interchange format file:

original:

    <http://darpa.mil/annotations/ldc/assertions/388>
        a               rdf:Statement ;
        rdf:object      ldcOnt:GeopoliticalEntity ;
        rdf:predicate   rdf:type ;
        rdf:subject     ldc:E781145.00381 ;
        aif:confidence  [ a                    aif:Confidence ;
                          aif:confidenceValue  "1.0"^^<http://www.w3.org/2001/XMLSchema#double> ;
                          aif:system           ldc:LDCModelGenerator
                        ] ;
        aif:system      ldc:LDCModelGenerator .

    ldc:E781145.00381  a  aif:Entity ;
            aif:system  ldc:LDCModelGenerator .

turned into:

    subj http://darpa.mil/annotations/ldc/assertions/388
    pred http://www.isi.edu/aida/interchangeOntology#confidence
    obj ub1bL12C25

    subj ub1bL12C25
    pred http://www.isi.edu/aida/interchangeOntology#system
    obj http://darpa.mil/annotations/ldc/LDCModelGenerator

    subj ub1bL12C25
    pred http://www.w3.org/1999/02/22-rdf-syntax-ns#type
    obj http://www.isi.edu/aida/interchangeOntology#Confidence

    subj http://darpa.mil/annotations/ldc/E781145.00381
    pred http://www.isi.edu/aida/interchangeOntology#system
    obj http://darpa.mil/annotations/ldc/LDCModelGenerator

    subj ub1bL12C25
    pred http://www.isi.edu/aida/interchangeOntology#confidenceValue
    obj 1.0

    subj http://darpa.mil/annotations/ldc/assertions/388
    pred http://www.w3.org/1999/02/22-rdf-syntax-ns#subject
    obj http://darpa.mil/annotations/ldc/E781145.00381

    subj http://darpa.mil/annotations/ldc/assertions/388
    pred http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate
    obj http://www.w3.org/1999/02/22-rdf-syntax-ns#type

    subj http://darpa.mil/annotations/ldc/E781145.00381
    pred http://www.w3.org/1999/02/22-rdf-syntax-ns#type
    obj http://www.isi.edu/aida/interchangeOntology#Entity

and so on.

so subj and obj are nodes, and pred is an edge label on the edge from subj to obj

Updated: Pengxiang Cheng, Aug 2020
- Small fix on cleanup and incorporating rdflib calls in the class.
"""

import logging
import os
from collections import defaultdict
from urllib.parse import urlsplit
from aida_utexas import util

import rdflib


class RDFNode:
    """RDFNode: as we are using a ttl-like notation, a RDFNode keeps
    all triples that share a subject, with the subject as the "node name".
    It keeps a dictionary "entry" with all preds as keys and the objects as
    values.
    """

    # initialization: remember the subj, and start an empty dict of pred/obj pairs
    def __init__(self, node_name):
        self.name = node_name
        self.out_edge = defaultdict(set)
        self.in_edge = defaultdict(set)

    # adding a pred/obj pair to a subj node
    def add_out_edge(self, pred, obj):
        self.out_edge[pred].add(obj)

    # adding a pred/subj pair to an obj node
    def add_in_edge(self, pred, subj):
        self.in_edge[pred].add(subj)

    # shorten any label by removing the URI path and keeping only the last bit
    @staticmethod
    def short_label(label):
        url_pieces = urlsplit(label)
        if url_pieces.fragment == '':
            return os.path.basename(url_pieces.path)
        else:
            return url_pieces.fragment

    # shorten the node name
    def short_name(self):
        return self.short_label(self.name)

    # write the node name, and all pred/object pairs, all in short form
    def prettyprint(self, omit=None):
        node_str = 'Node name: {}'.format(self.short_name())
        for pred, objs in self.out_edge.items():
            if omit is None or pred not in omit:
                node_str += '\n\t{}: {}'.format(pred, ' '.join(self.short_label(o) for o in objs))
        return node_str

    # given a pred, return the objs that go with it
    def get(self, pred, shorten=False):
        objs = self.out_edge.get(pred, set())
        if shorten:
            return set(self.short_label(o) for o in objs)
        else:
            return objs


class RDFGraph:
    # initialization builds an empty graph
    def __init__(self, node_cls=RDFNode):
        self.node_dict = {}
        self.node_cls = node_cls

    # # build the RDF graph from a file, default format is ttl (turtle)
    # def build_graph(self, graph_path, fmt='ttl'):
    #     # load triples from the file
    #     logging.info('Loading RDF graph from {} ...'.format(graph_path))

    #     # KE Feb 2022: Changed build_graph so that it can build a graph
    #     # from multiple ttl files at once
    #     rdf_filepaths = util.get_file_list(graph_path, suffix=fmt)

    #     for rdf_filepath in rdf_filepaths:
    #         graph = rdflib.Graph()
    #         graph.parse(str(rdf_filepath), format=fmt)

    #         logging.info('Done. Found {} triples.'.format(len(graph)))

    #         # for each new triple, record it
    #         logging.info('Building RDF graph ...')

    #         for subj, pred, obj in graph:
    #             # get the short label for predicate
    #             pred = RDFNode.short_label(pred)

    #             if subj not in self.node_dict:
    #                 self.node_dict[subj] = self.node_cls(subj)

    #             if obj not in self.node_dict:
    #                 self.node_dict[obj] = self.node_cls(obj)

    #             self.node_dict[subj].add_out_edge(pred, obj)
    #             self.node_dict[obj].add_in_edge(pred, subj)

    #     logging.info('Done.')
    
    # Build the RDF graph from a file, default format is ttl (turtle)
    def build_graph(self, graph_path, fmt='ttl'):
        # load triples from the file
        logging.info('Loading RDF graph from {} ...'.format(graph_path))

        graph = rdflib.Graph()
        graph.parse(graph_path, format=fmt)

        logging.info('Done. Found {} triples.'.format(len(graph)))

        # for each new triple, record it
        logging.info('Building RDF graph ...')

        for subj, pred, obj in graph:
            # get the short label for predicate
            pred = RDFNode.short_label(pred)

            if subj not in self.node_dict:
                self.node_dict[subj] = self.node_cls(subj)

            if obj not in self.node_dict:
                self.node_dict[obj] = self.node_cls(obj)

            self.node_dict[subj].add_out_edge(pred, obj)
            self.node_dict[obj].add_in_edge(pred, subj)

        logging.info('Done.')
        
    # printing out the graph in readable form
    def prettyprint(self, max_nodes=None):
        graph_str = ''

        node_count = 0
        for node in self.node_dict.values():
            graph_str += '\n\n{}\n{}'.format('=' * 14, node.prettyprint())
            node_count += 1
            if max_nodes and node_count >= max_nodes:
                break

        return graph_str
