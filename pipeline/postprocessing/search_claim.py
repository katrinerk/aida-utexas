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

from aida_utexas.aif.aida_graph import AidaGraph
from aida_utexas.aif.json_graph import JsonGraph
from aida_utexas import util

###
# given a claim ID and a json graph, find the matching claim and its associated KEs,
# and find all statements that are either type statements of associated KEs or link two associated KEs
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

def match_query_doc(csvfile_path):
    filepath = Path(csvfile_path)
    
    record = pandas.read_csv(filepath)
    unique_premise = record['premise_id'].unique()
    #need to alter the filter
    frecord = record[(record.adjust_label == 'contradiction') | (record.adjust_label == 'entailment')]
    matchclaim = {}
    matchttl = {}
    for premise in unique_premise:
        matchclaim[premise] = []
        for row in frecord.itertuples(index=True, name='Pandas'):
            if row.premise_id == premise:
                if row.hypo_id not in matchclaim[premise]:
                    matchclaim[premise].append(row.hypo_id) # query claimid -> doc claimid
                    matchttl[row.hypo_id]= row.hypo_ttl # doc claimid -> doc turtle file name
                    
    return (matchclaim, matchttl)  

def write_claim_file(claim_id, json_graph, file_path):
    for claim_label, claim_entry in json_graph.each_claim():
        if claim_entry.claimid == claim_id :
            #print(claim_label, claim_entry.text)
            node = json_graph.node_dict[claim_label]
            
            f = open('/Users/cookie/Downloads/{}.ttl'.format(node.claim_id),"w")
            f.write("ex:" + node.claim_id +  " a aida:" + node.type + "; \n")
            f.write("\taida:associatedKEs ")
            length = len(node.associated_kes)
            for ke in node.associated_kes:
                f.write("\t\t<" + ke + ">")
                length -= 1
                if length > 0:
                    f.write(",\n")
                else:
                    f.write(" ;\n")
           
            f.close()
            
            #print(node.date_time)
    
            
def main():
    
    parser = ArgumentParser()
    parser.add_argument('relative_file_path', help='Path to gunjan csv files')
    parser.add_argument('adjust_label_file_path', help='Path to yaling csv files')
    parser.add_argument('original_turtle_file_path', help='Path to original turtle files')  
    parser.add_argument('output_dir', help='Directory to write output files. turtle files will be in a subdirectory called the relative query claim id')
    
    args = parser.parse_args()
    doc_out_dir = util.get_output_dir(args.output_dir, overwrite_warning=not args.force)
    
    match_claims = match_query_doc(args.relative_file_path)[0]
    match_ttls = match_query_doc(args.relative_file_path)[1]
    
    for query_claim in match_claims.keys():
        for relative_doc_claim in match_claims[query_claim]:
            output_path = doc_out_dir / query_claim / "{}.ttl".format(relative_doc_claim)
            turtle_path = args.relative_file_path / match_ttls[relative_doc_claim]
            filepath = Path(turtle_path)
            aif_graph = AidaGraph()
            aif_graph.build_graph(str(filepath), fmt = "ttl")
            json_graph = JsonGraph()
            json_graph.build_graph(aif_graph)
            write_claim_file(relative_doc_claim, json_graph, output_path)
            #need to add stmts of associated KEs
            

if __name__ == '__main__':
    main()