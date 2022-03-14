# Code by Pengxiang Cheng
# Attempt to adapt to claim output: Katrin Erk

import sys
import os
import pandas

# find relative path_jy
from pathlib import Path
currpath = Path(os.getcwd())

sys.path.insert(0, str(currpath.parents[1]))


import math
from argparse import ArgumentParser
from io import BytesIO
from operator import itemgetter

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
# determine set of all clusters relevant for hte given EREs,
# and for each cluster prototype, determine handle
# return mapping from prototype to handles
def compute_handle_mapping(ere_set, json_graph, member_to_clusters, cluster_to_prototype):
    entity_cluster_set = set()
    for ere in ere_set:
        # Only include handles for clusters of Entities.
        if json_graph.is_entity(ere):
            for cluster in member_to_clusters[ere]:
                entity_cluster_set.add(cluster)

    proto_handles = {}
    for cluster in entity_cluster_set:
        prototype = cluster_to_prototype.get(cluster, None)
        if prototype is not None:
            proto_handles[prototype] = json_graph.node_dict[cluster].handle

    return proto_handles

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
    
    # add relevant query claims
    if claim_name in relevant_query.keys():
        rpred = URIRef(AIDA + 'relatedClaims')
        for query in relevant_query[claim_name]:   
            #obj = URIRef(EX + query)
            obj = Literal(query, datatype=XSD.string)
            all_triples.add((claim_id, rpred, obj))
    
    # add supporting query claims
    if claim_name in supporting_query.keys(): 
        spred = URIRef(AIDA + 'supportingClaims')
        for query in supporting_query[claim_name]:   
            #obj = URIRef(EX + query)
            obj = Literal(query, datatype=XSD.string)
            all_triples.add((claim_id, spred, obj))
    
    # add refuting query claims
    if claim_name in refuting_query.keys(): 
        rpred = URIRef(AIDA + 'refutingClaims')
        for query in refuting_query[claim_name]:   
            #obj = URIRef(EX + query)
            obj = Literal(query, datatype=XSD.string)
            all_triples.add((claim_id, rpred, obj))     

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

def main():
    parser = ArgumentParser()   
    parser.add_argument('relative_file_path', help='Path to gunjan csv files')
    parser.add_argument('suppporting_refuting_file_path', help='Path to yaling csv files')
    #parser.add_argument('claim_id', help="ID of the claim to keep")
    #parser.add_argument('filename', help="name of file to use, prefix of both ttl and json filename")
    parser.add_argument('graph_path', help='path to the graph json file') # single big json file
    parser.add_argument('kb_path', help='path to AIF file') # single big file
    parser.add_argument('output_dir', help='path to output directory')
    parser.add_argument('run_id', help='TA3 run ID')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')
    args = parser.parse_args()

    query_relevant_doc_claim, doc_claim_relevant_query, relevant_claim_ttl = relevant_file_filter(args.relative_file_path)
    doc_claim_match_refuting_query, doc_claim_match_supporting_query, ct_rf_claim_ttl = conflict_supporting_file_filter(args.suppporting_refuting_file_path)
    
    json_path = args.graph_path
    json_graph = JsonGraph.from_dict(util.read_json_file(json_path, 'JSON graph'))
    
    ###
    # identify ttl file: can be buried more deeply somewhere under kb_path
    kb_path = args.kb_path
    if kb_path is None:
        print("Error: KB not found", kb_path)
        sys.exit(1)

    # make mappings between clusters, cluster members, and prototypes
    # graph_mappings = json_graph.build_cluster_member_mappings()

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
    # jy: todo: kb_path should be just str, no need to be a path. need to test again
    kb_graph.parse(kb_path, format='ttl')
    
    run_id = args.run_id
    
    print("dealing single claim now \n")
    os.makedirs(Path(args.output_dir))
    for query in query_relevant_doc_claim.keys():

        
        #####
        # json file: extract associated KEs, and statements linking associated KEs
        for relevant_claim in query_relevant_doc_claim[query]:
            # from the json file, extract the correct claim and its associated KEs
            # associated_kes: a list of labels of knowledge elements, both sameas clusters and prototypes
            # associated_stmts: one type statement per KE, plus other statements as long as they connect
            #        two associated KEs
        
            material_dict = find_claim_associated_kes(relevant_claim, json_graph)
            if material_dict is None:
                print("Error: couldn't find claim", relevant_claim)
                continue
                #sys.exit(1)

            subgraph = build_subgraph_for_claim(material_dict, kb_graph, json_graph, relevant_claim, doc_claim_match_supporting_query, doc_claim_match_refuting_query, doc_claim_relevant_query)
            
            output_path = os.path.join(str(args.output_dir) + '/', relevant_claim + ".ttl")
            with open(output_path, 'w') as fout:
                fout.write(print_graph(subgraph))
                


if __name__ == '__main__':
    main()