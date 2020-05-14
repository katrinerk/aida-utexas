"""RDF graph:
takes as input an rdflib Graph
generated from an AIDA interchange format file
(the AIDA interchange format is based on the GAIA proposal)

Transforms the graph into the following format:
one node for each subject of the rdflib Graph
an entry for each predicate/object pair associated with this subject
in the rdflib graph

Here is how rdflib reads an AIDA interchange format file:

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
"""

import os
import urllib
from collections import defaultdict


class RDFNode:
    """RDFNode: as we are using a ttl-like notation, a RDFNode keeps
    all triples that share a subject, with the subject as the "node name".
    It keeps a dictionary "entry" with all preds as keys and the objects as
    values.
    """

    # initialization: remember the subj, and start an empty dict of pred/obj pairs
    def __init__(self, nodename):
        self.name = nodename
        self.outedge = defaultdict(set)
        self.inedge = defaultdict(set)

    # adding a pred/obj pair
    def add(self, pred, obj):
        self.outedge[pred].add(obj)

    # adding a pred/subj pair
    def add_inedge(self, pred, subj):
        self.inedge[pred].add(subj)

    # shorten any label by removing the URI path and keeping only the last bit
    @staticmethod
    def shortlabel(label):
        urlpieces = urllib.parse.urlsplit(label)
        if urlpieces.fragment == "":
            return os.path.basename(urlpieces.path)
        else:
            return urlpieces.fragment

    # shorten the node name
    def shortname(self):
        return self.shortlabel(self.name)

    # prettyprint: write the node name, and all pred/object pairs, all in short form
    def prettyprint(self, omit=None):
        print("Node name", self.shortname())
        for pred, obj in self.outedge.items():
            if omit is None or self.shortlabel(pred) not in omit:
                print("\t", self.shortlabel(pred), ":", " ".join(self.shortlabel(o) for o in obj))

    # get: given a pred, return the obj's that go with it
    def get(self, targetpred, shorten=False):
        if targetpred in self.outedge:
            return self._maybe_shorten(self.outedge[targetpred], shorten)
        else:
            for pred in self.outedge.keys():
                if self.shortlabel(pred) == targetpred:
                    return self._maybe_shorten(self.outedge[pred], shorten)
        return set([])

    def _maybe_shorten(self, labellist, shorten=False):
        if shorten:
            return set([self.shortlabel(l) for l in labellist])
        else:
            return set(labellist)


class RDFGraph:
    # initialization builds an empty graph
    def __init__(self, nodeclass=RDFNode):
        self.node_dict = {}
        self.nodeclass = nodeclass

    # adding another RDF file to the graph of this object
    def add_graph(self, rdflibgraph):
        # for each new triple, record it
        for subj, pred, obj in rdflibgraph:
            
            if subj not in self.node_dict:
                self.node_dict[subj] = self.nodeclass(subj)

            if obj not in self.node_dict:
                self.node_dict[obj] = self.nodeclass(obj)
                
            self.node_dict[subj].add(pred, obj)
            self.node_dict[obj].add_inedge(pred, subj)

    # printing out the graph in readable form
    def prettyprint(self):
        for node in self.node_dict.values():
            print("==============")
            node.prettyprint()
