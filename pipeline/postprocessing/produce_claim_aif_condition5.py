# Code by Pengxiang Cheng
# Attempt to adapt to claim output: Katrin Erk
# test jy

import sys
import os
import pandas
import logging

# find relative path_jy
from pathlib import Path
currpath = Path(os.getcwd())

sys.path.insert(0, str(currpath.parents[1]))


import math
from argparse import ArgumentParser
from io import BytesIO
from operator import itemgetter
from collections import defaultdict
import csv

import rdflib
from rdflib import Graph
from rdflib.namespace import Namespace, RDF, XSD, NamespaceManager
from rdflib.plugins.serializers.turtle import TurtleSerializer
from rdflib.plugins.serializers.turtle import VERB
from rdflib.term import BNode, Literal, URIRef

from aida_utexas import util
from aida_utexas.aif import JsonGraph
from pipeline.rdflib_helper import AIDA, LDC, LDC_ONT, EX
from pipeline.rdflib_helper import catalogue_kb_nodes
from pipeline.rdflib_helper import index_statement_nodes, index_cluster_membership_nodes
from pipeline.rdflib_helper import index_type_statement_nodes
from pipeline.rdflib_helper import triples_for_cluster, triples_for_cluster_membership, triples_for_claim
from pipeline.rdflib_helper import triples_for_ere, triples_for_type_stmt, triples_for_edge_stmt

UTEXAS = Namespace('http://www.utexas.edu/aida/')

# trying to match the AIF format
class AIFSerializer(TurtleSerializer):
    xsd_namespace_manager = NamespaceManager(Graph())
    xsd_namespace_manager.bind('xsd', XSD)

    # when writing BNode as subjects, write closing bracket
    # in a new line at the end
    def s_squared(self, subject):
        if (self._references[subject] > 0) or not isinstance(subject, BNode):
            return False
        self.write('\n' + self.indent() + '[')
        self.predicateList(subject)
        self.write('\n] .')
        return True

    # when printing Literals, directly call Literal.n3()
    def label(self, node, position):
        if node == RDF.nil:
            return '()'
        if position is VERB and node in self.keywords:
            return self.keywords[node]
        if isinstance(node, Literal):
            return node.n3(namespace_manager=self.xsd_namespace_manager)
        else:
            node = self.relativize(node)

            return self.getQName(node, position == VERB) or node.n3()

#################################
# return the text representation of the graph
def print_graph(g):
    serializer = AIFSerializer(g)
    stream = BytesIO()
    serializer.serialize(stream=stream)
    return stream.getvalue().decode()

#############################
# NEW
# (or rather updated)
# determine set of all clusters relevant for hte given EREs,
# and for each cluster prototype, determine handle
# return mapping from prototype to handles
def compute_handle_mapping(ere_set, json_graph, member_to_clusters, cluster_to_prototype):
    entity_cluster_set = set()
    cluster_labels = defaultdict(list)
    
    for ere in ere_set:
        # Only include handles for clusters of Entities.
        if json_graph.is_entity(ere):
            for cluster in member_to_clusters[ere]:
                entity_cluster_set.add(cluster)
                for erename in json_graph.node_dict[ere].name:
                    if len(erename) > 0:
                        cluster_labels[cluster].append(erename)

    proto_handles = {}
    for cluster in entity_cluster_set:
        prototype = cluster_to_prototype.get(cluster, None)
        if prototype is not None:
            firsthandle = json_graph.node_dict[cluster].handle
            if firsthandle != "":
                # handle on the prototype is there and is not empty
                proto_handles[prototype] = firsthandle
            else:
                # any better handle to be had?
                if len(cluster_labels[cluster]) > 0:
                    proto_handles[prototype] = cluster_labels[cluster][0]
                else:
                    # nope, nothing better there
                    proto_handles[prototype] = firsthandle
                    

    return proto_handles

#########3
# NEW compute info on clusters needed for handling prototype handles
# returns: ere_set, member_to_clusters, cluster_to_prototype 
def compute_cluster_info(json_graph, ere_list):
    graph_mappings = json_graph.build_cluster_member_mappings()
    member_to_clusters = graph_mappings['member_to_clusters']
    cluster_to_prototype = graph_mappings['cluster_to_prototype']

    return (set(ere_list), member_to_clusters, cluster_to_prototype)

#########
# given a claim ID and a json graph, find the matching claim and its associated KEs,
# and find all statements that are either type statements of associated KEs or link two associated KEs.
# also include the claim ID itself in the list of associated KEs
# returns a dictionary with entries
# 'edge_stms',
# 'type_stmts',
# 'claim',
# 'eres'
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

#############################
def build_subgraph_for_claim(material_dict, kb_graph, json_graph, claim_name, supporting_query, refuting_query, relevant_query):

    ##########
    # make collection of all triples that need to go into the subgraph

    # All triples to be added to the subgraph
    # logging.info('Extracting all content triples')
    all_triples = set()

    # adding a single claim
    claim_id = URIRef(material_dict["claim"])
    all_triples.update(triples_for_claim(kb_graph, claim_id))

    # EREs
    for ere in material_dict["eres"]:
        kb_ere_id = URIRef(ere)
        all_triples.update(triples_for_ere(kb_graph, kb_ere_id))

    # type statements
    for kb_stmt_id in material_dict['type_stmts']:
        #jy
        t_stmt_id = URIRef(kb_stmt_id)
        all_triples.update(triples_for_type_stmt(kb_graph, t_stmt_id))

    # edge statements
    for kb_stmt_id in material_dict["edge_stmts"]:
        #jy
        e_stmt_id = URIRef(kb_stmt_id)
        all_triples.update(triples_for_edge_stmt(kb_graph, e_stmt_id))
        
    ### jy
    # add relevant/supporting/refuting query claims to doc claims for condition 5
    # try to add related/support claims to that claim in the rdflib
    
    # add supporting query claims
    marked_claims = set()
    if claim_name in supporting_query.keys(): 
        spred = URIRef(AIDA + 'supportingClaims')
        for claim2 in supporting_query[claim_name]: 
            if claim2 in marked_claims:
                continue  
            else:
                #obj = URIRef(EX + query)
                obj = Literal(claim2, datatype=XSD.string)
                all_triples.add((claim_id, spred, obj))
                marked_claims.add(claim2)
    
    # add refuting query claims
    if claim_name in refuting_query.keys(): 
        rpred = URIRef(AIDA + 'refutingClaims')
        for claim2 in refuting_query[claim_name]: 
            if claim2 in marked_claims:
                continue  
            else:  
                #obj = URIRef(EX + query)
                obj = Literal(claim2, datatype=XSD.string)
                all_triples.add((claim_id, rpred, obj)) 
                marked_claims.add(claim2) 
            
    # add relevant query claims
    if claim_name in relevant_query.keys():
        rpred = URIRef(AIDA + 'relatedClaims')
        for claim2 in relevant_query[claim_name]: 
            if claim2 in marked_claims:
                continue  
            else:    
                #obj = URIRef(EX + query)
                obj = Literal(claim2, datatype=XSD.string)
                all_triples.add((claim_id, rpred, obj))
                marked_claims.add(claim2)     

    #############3
    # logging.info('Constructing a subgraph')
    # Start building the subgraph
    subgraph = Graph()

    # Bind all prefixes of kb_graph to the subgraph
    for prefix, namespace in kb_graph.namespaces():
        if str(namespace) not in [AIDA, LDC, LDC_ONT]:
            subgraph.bind(prefix, namespace)
    # Bind the AIDA, LDC, LDC_ONT, and UTEXAS namespaces to the subgraph
    # jy: may need to delete these unnecessary prefixes 02/20
    subgraph.bind('aida', AIDA, override=True)
    subgraph.bind('ldc', LDC, override=True)
    subgraph.bind('ldcOnt', LDC_ONT, override=True)
    subgraph.bind('utexas', UTEXAS)
    subgraph.bind('ex', EX)


    # logging.info('Adding all content triples to the subgraph')
    # Add all triples
    for triple in all_triples:
        subgraph.add(triple)

    # NEW
    # fix prototype handles
    # Compute cluster info
    ere_set, member_to_clusters, cluster_to_prototype = compute_cluster_info(json_graph, material_dict["eres"])

    # Compute handles for Entity clusters
    proto_handles = compute_handle_mapping(
        ere_set, json_graph, member_to_clusters, cluster_to_prototype)

    # add in handles for prototypes where they are missing
    for proto_ere, handle in proto_handles.items():
        kb_proto_id = URIRef(proto_ere)
        if len(list(subgraph.objects(subject=kb_proto_id, predicate=AIDA.handle))) == 0:
            subgraph.add((kb_proto_id, AIDA.handle, Literal(handle, datatype=XSD.string)))
    
    return subgraph

###
# given a directory,
# walk all subdirectories until we find a file with the given name
def find_ttl_file(startpath, filename):
    for root, dirs, filenames in os.walk(startpath):
        if filename in filenames:
            fullname = os.path.join(root, filename)
            return fullname
    return None
   
### jy
# return supporting/refuting relations between queries and doc claims
def conflict_supporting_file_filter(csvfile_path):
    filepath = Path(csvfile_path)
    
    record = pandas.read_csv(filepath)
    unique_premise = record['Query_ID'].unique()
    hypo_match_supporting_premise = {}
    hypo_match_refuting_premise = {}
    claim_ttl = {}
    
    for premise in unique_premise:
        ctdrecord = record[record.nli_label == 'contradiction']
        for row in ctdrecord.itertuples(index=True, name='Pandas'):
            if row.Query_ID == premise:
                claim_ttl[row.Claim_ID]= row.Claim_Filename # doc claimid -> doc turtle file name
                if row.Claim_ID not in hypo_match_refuting_premise.keys():
                    hypo_match_refuting_premise[row.Claim_ID] = [] 
                if premise not in hypo_match_refuting_premise[row.Claim_ID]:
                    hypo_match_refuting_premise[row.Claim_ID].append(premise) # doc claimid -> refuting query claimids
     
        rfrecord = record[record.nli_label == 'entailment']
        for row in rfrecord.itertuples(index=True, name='Pandas'):
            if row.premise_id == premise:
                claim_ttl[row.Claim_ID]= row.Claim_Filename # doc claimid -> doc turtle file name
                if row.Claim_ID not in hypo_match_supporting_premise.keys():
                    hypo_match_supporting_premise[row.Claim_ID] = []
                if premise not in hypo_match_supporting_premise[row.Claim_ID]:
                    hypo_match_supporting_premise[row.Claim_ID].append(premise) # doc claimid -> supporting query claimids
                         
    return (hypo_match_refuting_premise, hypo_match_supporting_premise, claim_ttl)  

### jy
# return relevant relations between queries and doc claims
def relevant_file_filter(csvfile_path):
    filepath = Path(csvfile_path)       
    record = pandas.read_csv(filepath)

    unique_premise = record['Query_ID'].unique()
    #hot fix to remove spaces from column names
    record.rename(columns = {'Related or Unrelated': 'Related_or_Unrelated'}, inplace=True)
    frecord = record[record.Related_or_Unrelated == 'Related']
    
    premise_match_hypo = {}
    hypo_match_relevant_premise = {}
    claim_ttl = {}
    for premise in unique_premise:
        premise_match_hypo[premise] = []
        for row in frecord.itertuples(index=True, name='Pandas'):
            if row.Query_ID == premise:
                if row.Claim_ID not in premise_match_hypo[premise]: #remove duplicate query-doc pairs
                    premise_match_hypo[premise].append(row.Claim_ID) # query claimid -> doc claimid
                    claim_ttl[row.Claim_ID]= row.Claim_Filename # doc claimid -> doc turtle file name
                if row.Claim_ID not in hypo_match_relevant_premise.keys():
                    hypo_match_relevant_premise[row.Claim_ID] = [] 
                if premise not in hypo_match_relevant_premise[row.Claim_ID]:
                    hypo_match_relevant_premise[row.Claim_ID].append(premise) # doc claimid -> refuting query claimids
            
    return (premise_match_hypo, hypo_match_relevant_premise, claim_ttl)

# make a ranking for a given query
def make_ranking(query_id, query_claim_score, claim_claim_score, query_2_text, claim_2_text):
    
    # determine the highest-ranked claim
    query_claims = query_claim_score[query_id].keys()

    if len(query_claims) == 0:
        return [], {}
    
    max_claim = sorted(query_claims, key = lambda claim:query_claim_score[query_id][claim], reverse = True)[0]
        
    claims_ranked = [ max_claim ]
    claim_score = { }
    claim_score[ max_claim] = query_claim_score[query_id][max_claim]
    
    while len(claims_ranked) < len(query_claims):
        summed_score = { }

        # for each claim that is not yet ranked:
        # determine summed score with all already ranked claims
        for claim1 in query_claims:
            if claim1 in claims_ranked: continue

            summed_score[claim1] = 0
            for claim2 in claims_ranked:
                
                if (claim1, claim2) not in claim_claim_score:
                    print("Error, missing score for", claim1, claim2)
                    sys.exit(0)

                summed_score[claim1] += claim_claim_score[ (claim1, claim2) ]

        # include the claim with minimum summed score, minimum relatedness to other claims
        min_claim = sorted(summed_score.keys(), key = lambda claim:summed_score[claim])[0]
            
        claims_ranked.append(min_claim)
        claim_score[min_claim] = summed_score[min_claim]

    return claims_ranked, claim_score

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

def main():
    parser = ArgumentParser()   
    parser.add_argument('working_dir', help="Working directory with intermediate system results")
    parser.add_argument('run_id', help="run ID, same as subdirectory of Working")
    parser.add_argument('condition', help="Condition5, Condition6, Condition7")
    
    #parser.add_argument('relative_file_path', help='Path to gunjan csv files') #need to rename
    parser.add_argument('suppporting_refuting_file_path', help='Path to yaling csv files') # need to rename 
    
    parser.add_argument('graph_path', help='path to the graph json file') # single big json file
    parser.add_argument('kb_path', help='path to AIF file') # single big file
    
    parser.add_argument('output_dir', help='path to output directory')

    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')
    args = parser.parse_args()
    
    # sanity check on condition
    if args.condition not in ["Condition5", "Condition6", "Condition7"]:
        print("Error: need a condition that is Condition5, Condition6, Condition7")
        sys.exit(1)

    working_mainpath = util.get_input_path(args.working_dir)
    working_path = util.get_input_path(working_mainpath / args.run_id / args.condition) 
    
    #filter bad claims without associated KE that is
    # an event or relation with at least one edge that also goes to an associated KE
    json_path = args.graph_path
    json_graph = JsonGraph.from_dict(util.read_json_file(json_path, 'JSON graph'))

    
    all_claims = []
    #for condition 5
    #all claims can be extracted from q2d_relatedness.csv
    filename= util.get_input_path(working_path / "Step1_query_claim_relatedness" / "q2d_relatedness.csv")
    with open(str(filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            claim_id = row["Claim_ID"]
            all_claims.append(claim_id)
    
    clusterinfo = get_cluster_info_forgraph(json_graph)
    
    good_claims = []
     
    for claim in all_claims:
        claim_is_okay = test_claim_has_oneedge(json_graph, claim, clusterinfo)
        if claim_is_okay:
            good_claims.append(claim)
        else:
            print("bad claim: {}".format(claim))
    
    ### ranking processing
    # for each query, determine the list of related claims that match in topic/subtopic/template
    # this is in [working_path] / step3_claim_claim_ranking / query_claim_matching.csv
    query_related = defaultdict(list)
    claim_related = defaultdict(list)

    filename= util.get_input_path(working_path / "Step3_claim_claim_ranking" / "query_claim_matching.csv")
    
    with open(str(filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query_id = row["Query_ID"]
            claim_id = row["Claim_ID"]
            # filter bad claim
            if claim_id in good_claims:
                query_related[ query_id ].append(claim_id)
                claim_related[ claim_id ].append(query_id)

    # if condition5, read supporting/refuting/related relations of claims to queries

    if args.condition == "Condition5":
        query_claim_relation = { }

        # claims that are related and matching: located in [working_path]/step2_query_claim_nli/q2d_nli.csv
        queryclaim_rel_filename = util.get_input_path(working_path / "Step2_query_claim_nli" / "q2d_nli.csv")
    
        with open(str(queryclaim_rel_filename), newline='') as csvfile:
            # first row has header, so we don't need to give fieldnames
            reader = csv.DictReader(csvfile)
            for row in reader:
                query_id = row["Query_ID"]
                claim_id = row["Claim_ID"]
                nli_label = row["nli_label"]
                nist_label = {"neutral" : "related", "contradiction" : "refuting", "entailment" : "supporting"}[nli_label]

                query_claim_relation[ (query_id, claim_id) ] = nist_label
    else:
        query_claim_relation = { }

    # read relatedness ratings for claim pairs
    # this is in [working_path] / step3_claim_claim_ranking / claim_claim_redundancy.csv

    claim_claim_score = { }

    filename= util.get_input_path(working_path / "Step3_claim_claim_ranking" / "claim_claim_redundancy.csv")
    
    with open(str(filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            claim1 = row["Claim1_ID"]
            claim2 = row["Claim2_ID"]
            score = row["Score"]
            #filter bad claim
            if claim1 in good_claims and claim2 in good_claims:
                claim_claim_score[ (claim1, claim2) ] = float(score)
                claim_claim_score[ (claim2, claim1) ] = float(score)

    # read relatedness ratings between query and claims
    # this is in [working_path] / step1_query_claim_relatedness / q2d_relatedness.csv
    query_claim_score = defaultdict(dict)
    query_2_text = { }
    claim_2_text = { }

    filename= util.get_input_path(working_path / "Step1_query_claim_relatedness" / "q2d_relatedness.csv")
    
    with open(str(filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query_id = row["Query_ID"]
            query_text = row["Query_Sentence"]
            claim_id = row["Claim_ID"]
            claim_text = row["Claim_Sentence"]
            score = row["Score"]

            if claim_id in good_claims and claim_id in query_related[ query_id ]:
                query_claim_score[ query_id][claim_id]  = float(score)

            query_2_text[ query_id ] = query_text
            claim_2_text[ claim_id ] = claim_text
        
    # for each query, produce a ranking file named
    # QueryID.ranking.tsv.
    # Fields:
    # Condition5: Query_ID Claim_ID Rank Relation_to_Query
    # Condition6: Query_ID Claim_ID Rank
    # Condition7: Query_ID Claim_ID Rank

    # output is to: [working_path] / step4_ranking
    #jy:update
    output_path = util.get_output_dir(str(args.output_dir + '/' +  "out" + '/' + args.run_id + '/' + args.condition), overwrite_warning=not args.force)

    for query_id in query_related.keys():
        #jy
        output_dir = Path(str(output_path) + '/{}'.format(query_id))
        os.makedirs(output_dir)
        # make a ranking
        ranked_claims, claim_scores = make_ranking(query_id, query_claim_score, claim_claim_score, query_2_text, claim_2_text)
        
        # print("Sanity check")
        # print("Query", query_id, query_2_text[query_id], "\n")
        # for claim in ranked_claims:
        #     print(claim, claim_scores[claim], claim_2_text[claim])

        # and write it to a file
        output_filename = output_dir / (query_id + ".ranking.tsv")

        # Condition5: write including relation to query
        if args.condition == "Condition5":

            with open(output_filename, 'w', newline='') as csvfile:
                #fieldnames = ['Query_ID', 'Claim_ID', 'Rank', 'Relation_to_Query']
                writer = csv.writer(csvfile, delimiter="\t")

                for rank, claim_id in enumerate(ranked_claims):
                    rel = query_claim_relation.get( (query_id, claim_id), "related")
                    writer.writerow( [ query_id, claim_id, rank + 1, rel ])

        # Conditions 6, 7: write without relation to query
        else:
            with open(output_filename, 'w', newline='') as csvfile:
                fieldnames = ['Query_ID', 'Claim_ID', 'Rank']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")

                writer.writeheader()
                for rank, claim_id in enumerate(ranked_claims):
                    writer.writerow( { "Query_ID" : query_id, "Claim_ID" : claim_id, "Rank" : rank + 1})
                    
    
    ### turtle files output processing
    #query_relevant_doc_claim, doc_claim_relevant_query, relevant_claim_ttl = relevant_file_filter(args.relative_file_path)
    doc_claim_match_refuting_query, doc_claim_match_supporting_query, ct_rf_claim_ttl = conflict_supporting_file_filter(args.suppporting_refuting_file_path)
    
    ###
    # identify ttl file: can be buried more deeply somewhere under kb_path
    kb_path = args.kb_path
    if kb_path is None:
        print("Error: KB not found", kb_path)
        sys.exit(1)

    # TODO: there is a known bug in rdflib that
    #  rdflib.Literal("2008", datatype=rdflib.XSD.gYear) would be parsed into
    #  rdflib.term.Literal(u'2008-01-01', datatype=rdflib.XSD.gYear) automatically,
    #  because a `parse_date` function is invoked for all rdflib.XSD.gYear literals.
    #  This is a temporary workaround to patch the _toPythonMapping locally.
    #  c.f.: https://github.com/RDFLib/rdflib/issues/806
    # noinspection PyProtectedMember
    
    ###jy
    #rdflib.term._toPythonMapping.pop(rdflib.XSD['gYear'])

    print('Reading kb from {}'.format(kb_path))
    # rdflib graph object
    kb_graph = Graph()
    kb_graph.parse(kb_path, format='ttl')
    
    '''
    print("creating query directory \n")
    for query in query_relevant_doc_claim.keys():
        if query_relevant_doc_claim[query] == None:
            continue
        output_dir = Path(args.output_dir + '/{}'.format(query))
        os.makedirs(output_dir)
    
    output_dir = Path(args.output_dir)
    '''
    
    
    print("dealing single claim now \n")
    cnt = 0
    for claim in claim_related.keys():
        # from the json file, extract the correct claim and its associated KEs
        # associated_kes: a list of labels of knowledge elements, both sameas clusters and prototypes
        # associated_stmts: one type statement per KE, plus other statements as long as they connect
        #        two associated KEs
    
        material_dict = find_claim_associated_kes(claim, json_graph)
        if material_dict is None:
            print("Error: couldn't find claim", claim)
            sys.exit(1)    

        subgraph = build_subgraph_for_claim(material_dict, kb_graph, json_graph, claim, doc_claim_match_supporting_query, doc_claim_match_refuting_query, claim_related) 
        
        for query in claim_related[claim]: 
            file_path = os.path.join(str(str(output_path) + '/' + query + '/'), claim + ".ttl")
            with open(file_path, 'w') as fout:
                fout.write(print_graph(subgraph))
      


if __name__ == '__main__':
    main()