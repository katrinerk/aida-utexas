import sys
sys.path.insert(0, "/Users/kee252/Documents/Projects/AIDA/scripts/aida-utexas")


import io
import json
import logging
import sys
import os
from argparse import ArgumentParser
from collections import defaultdict
import csv
import json

from aida_utexas import util
from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis import AidaHypothesisCollection, AidaHypothesisFilter

###
# determine cluster info for the json graph
# run this once globally for the graph
# before running any test_claim_has_oneedge calls
def get_cluster_info_forgraph(json_graph):

    node2cluster = {} # store node label -> sameAsClusterNode
    clusterlabel = set() # store sameAsCluster node label
    cluster2members = defaultdict(list) # store cluster label -> members
    
    for node_label, node in json_graph.node_dict.items():
        if node.type == 'SameAsCluster':
            clusterlabel.add(node_label)
            node2cluster[node_label] = node
        elif node.type == "ClusterMembership":
            cluster2members[ node.cluster].append(node.clusterMember)


    return { "node2cluster" : node2cluster,
             "clusterlabels" : clusterlabel,
             "cluster2members" : cluster2members}


###
# Katrin Erk March 2022: given a json graph and a claim ID,
# test whether the claim has at least one associated KE that is
# an event or relation with at least one edge that also goes to an associated KE
#
# returns: True if test passed, else False
def test_claim_has_oneedge(json_graph, claim_id, clusterinfo):

    ###
    # find the claim
    this_claim_label = None
    this_claim_entry = None
    
    for claim_label, claim_entry in json_graph.each_claim():
        if str(claim_entry.claim_id) == claim_id:
            # found the right claim
            this_claim_label = claim_label
            this_claim_entry = claim_entry
            break

    if this_claim_label is None:
        # we didn't find the claim
        logging.warning(f'Test for edge presence in claim: claim not found: {claim_id}')
        return False

    ####
    # determine EREs that are associated KEs
    # EREs: associated KEs
    eres = set(this_claim_entry.associated_kes)

    # for KEs that are clusters, add prototypes and other members to list of EREs
    for ake in this_claim_entry.associated_kes:
        # if we got a cluster, find the prototype
        if str(ake) in clusterinfo["clusterlabels"]:
            eres.add(clusterinfo["node2cluster"][str(ake)].prototype)
            # and all members
            for member in clusterinfo["cluster2members"][ str(ake) ]:
                eres.add(member)


    ###
    # test whether there are associated KEs that are events or relations
    # that have at least one edge going to an associated KE
    evrel_with_edge = set()
    for ake in list(eres):
        if json_graph.is_event(ake) or json_graph.is_relation(ake):
            for stmt in json_graph.each_ere_adjacent_stmt(ake):
                if json_graph.is_type_stmt(stmt):
                    continue
                else:
                    obj = json_graph.stmt_object(stmt)
                    if obj in eres:
                        evrel_with_edge.add(ake)

    # return: True (test passed) if we have at least one event or relation with an edge,
    # else false
    return len(evrel_with_edge) > 0
    
#############
# main


claim_id = sys.argv[1]
graph_path = sys.argv[2]

json_graph = JsonGraph.from_dict(util.read_json_file(graph_path, 'JSON graph'))

clusterinfo = get_cluster_info_forgraph(json_graph)

claim_is_okay = test_claim_has_oneedge(json_graph, claim_id, clusterinfo)

print("claim", claim_id, "is okay?", claim_is_okay)
