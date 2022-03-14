import os
import sys

from pathlib import Path
currpath = Path(os.getcwd())

sys.path.insert(0, str(currpath.parents[1]))

import io
import json
import logging
from argparse import ArgumentParser
from collections import defaultdict

from aida_utexas.aif.aida_graph import AidaGraph, AidaNode
from aida_utexas.aif.json_graph import JsonGraph
from aida_utexas.aif.rdf_graph import RDFGraph, RDFNode


from aida_utexas import util
import rdflib


def find_claim_associated_kes(claim_id, json_graph):
 
    sac = {} # store node label -> sameAsClusterNode
    saclabel = [] # store sameAsCluster node label
    for node_label, node in json_graph.node_dict.items():
        if node.type == 'SameAsCluster':
            saclabel.append(node_label)
            sac[node_label] = node       

    for claim_label, claim_entry in json_graph.each_claim():
        if str(claim_entry.claim_id) == claim_id: #jy: type of claim_entry.claim_id should be converted to str
            # found the right associated KEs
            associated_kes = set(claim_entry.associated_kes)  #each ke is a rdflib.term.URIRef object

            # for KEs that are clusters, add prototypes
            for ake in claim_entry.associated_kes:
                #jy
                if str(ake) in saclabel:               
                    associated_kes.add(sac[str(ake)].prototype)
                print("{}cluster prototype: {} \n".format(ake, sac[str(ake)].prototype))          
            
            # determine associated statements
            associated_stmts = set() #each stmt is a rdflib.term.URIRef object
            
            for ake in list(associated_kes):
                # this time, skip clusters
                if str(ake) in saclabel: 
                    continue
                    
                #print("hier0", ake, json_graph.has_node(ake), list(json_graph.each_ere_adjacent_stmt(ake)))
                
                type_found = False
                for stmt in json_graph.each_ere_adjacent_stmt(ake):
                    
                    if json_graph.is_type_stmt(stmt):
                        # keep only one type statement 
                        if type_found: pass
                        else: 
                            type_found = True
                            associated_stmts.add(stmt)
                            print("type stmt: {}".format(stmt))
                    else:
                        # non-type-statement: check whether both adjacent are associated KEs
                        predsubj, predobj = json_graph.stmt_args(stmt)
                        if predsubj in associated_kes and predobj in associated_kes:
                            associated_stmts.add(stmt)
                            print("non-type stmt: {}".format(stmt))
        
        
            print("The following are kes: \n")
            for ke in associated_kes:
                print(ke + '\n')  
            print("The following are stms: \n") 
            for stmt in associated_stmts:
                print(stmt + '\n')   
                
            return (associated_kes, associated_stmts)
        
        else:
            return None
            
            
            
                    


