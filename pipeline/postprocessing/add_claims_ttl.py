import os
import sys

from pathlib import Path
import turtle
currpath = Path(os.getcwd())

sys.path.insert(0, str(currpath.parents[1]))

import io
import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
import pandas

from tqdm import tqdm

from aida_utexas.aif.aida_graph import AidaGraph, AidaNode
from aida_utexas.aif.json_graph import JsonGraph
from aida_utexas.aif.rdf_graph import RDFGraph, RDFNode

import rdflib
from rdflib import Graph
from aida_utexas import util
from rdflib.term import BNode, Literal, URIRef
from rdflib.namespace import Namespace, RDF, XSD, NamespaceManager
from urllib.parse import urlsplit

from aida_utexas.aif.aida_graph import AidaGraph
from aida_utexas.aif.json_graph import JsonGraph

'''
g = RDFGraph()
g.build_graph("/Users/cookie/Downloads/dry-run-ta1.20220127/L0C0492EZ.ttl", fmt='ttl')
nodes = g.node_dict.keys()
preds = []
cnode = []
KE = []
for node in nodes:
    for pred in g.node_dict[node].out_edge.keys():
            if pred == "type":
                #cnode.append(node)
                print(node)
                print("\n")
            #if pred == "associatedKEs":
                
for n in cnode:              
    g.remove_triple((n, None, None))
    g.remove_triple((None, n, None))
    g.remove_triple((None, None, n))

g.serialize('output.ttl', 'turtle') 
'''

'''
filepath = Path("/Users/cookie/Downloads/dry-run-ta1.20220127/L0C0492EZ.ttl")
aif_graph = AidaGraph()
aif_graph.build_graph(str(filepath), fmt = "ttl")
json_graph = JsonGraph()
json_graph.build_graph(aif_graph)
claim_id = 'claim_L0C04958D_1'
'''

'''
for claim_label, claim_entry in json_graph.each_claim():
    if claim == str(claim_entry.claim_id):
        print("found {}".format(claim))
'''

def find_claim_associated_kes(claim_id, json_graph):


    node2cluster = {} # store node label -> sameAsClusterNode
    clusterlabel = set() # store sameAsCluster node label
    for node_label, node in json_graph.node_dict.items():
        if node.type == 'SameAsCluster':
            clusterlabel.add(node_label)
            node2cluster[node_label] = node       
    
    retv = { }
    
    for claim_label, claim_entry in json_graph.each_claim():
        if str(claim_entry.claim_id) == claim_id:
            # found the right claim
            retv["claim"] = claim_label
            retv["edge_stmts"] = set()
            retv["type_stmts"] = set()
            
            
            # found the right associated KEs
            retv["eres"] = set(claim_entry.associated_kes)

            # for KEs that are clusters, add prototypes
            for ake in claim_entry.associated_kes:
                # if we got a cluster, find the prototype
                if str(ake) in clusterlabel:
                    retv["eres"].add(node2cluster[str(ake)].prototype)
            
            # determine associated statements
            
            for ake in list(retv["eres"]):
                # this time, skip clusters
                if str(ake) in clusterlabel:
                    continue
                    
                type_found = False
                for stmt in json_graph.each_ere_adjacent_stmt(ake):
                    if json_graph.is_type_stmt(stmt):
                        # keep only one type statement 
                        if type_found: pass
                        else: 
                            type_found = True
                            retv["type_stmts"].add(stmt)
                            
                    else:
                        # non-type-statement: check whether both adjacent are associated KEs
                        predsubj, predobj = json_graph.stmt_args(stmt)
                        if predsubj in retv["eres"] and predobj in retv["eres"]:
                            retv["edge_stmts"].add(stmt)
                            
            return retv
        
    return None


    
    
    
def main():
    
    '''
        # get rdflib graph of the original ttl file
    kb_graph = Graph()
    filepath = Path("/Users/cookie/Downloads/dry-run-ta1.20220127/L0C0492EZ.ttl")
    kb_graph.parse("/Users/cookie/Downloads/dry-run-ta1.20220127/L0C0492EZ.ttl", format = 'ttl')

    # get json graph of that ttl file 
    # get claimid 
    aif_graph = AidaGraph()
    aif_graph.build_graph(str(filepath), fmt = "ttl")
    json_graph = JsonGraph()
    json_graph.build_graph(aif_graph)
    claim = 'claim_L0C04958D_1'
    material_dict = find_claim_associated_kes(claim, json_graph)


    # try to add related/support claims to that claim in the rdflib
    claim_id = URIRef(material_dict["claim"])
    print(claim_id)
    
    #for s, p, o in kb_graph.triples((claim_id, None, None)):
    #    print ("{}, {}, {}\n".format(s, p, o))
    
    aidaPrefix = 'https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#'
    pred = URIRef(aidaPrefix + 'related_claims')
    obj1 = Literal("claim-CLL0C04979A")
    #kb_graph.namespace_manager.bind('xsd', XSD)
    
    obj1 = Literal("claim-CLL0C04979A", datatype=XSD.string)
    #obj1.n3(kb_graph.namespace_manager)
    kb_graph.add((claim_id, pred, obj1))
    
    obj2 = Literal("claim-CLL0C04979A_2", datatype=XSD.string)
    kb_graph.add((claim_id, pred, obj2))
    

    # serilize it   
    kb_graph.serialize(destination='output.ttl', format='turtle')
    
    '''
    
    year = Literal("2021-01-01", datatype=XSD.gYear)
    tmp_year = str(year)
    print(tmp_year)
    list = tmp_year.split('-')
    print(list[0])
    new_year = Literal(list[0], datatype=XSD.gYear)
    print(type(new_year))

    

if __name__ == '__main__':
    main()             


