# Code by Pengxiang Cheng
# Attempt to adapt to claim output: Katrin Erk

import sys
import os

from collections import defaultdict

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
from pipeline.rdflib_helper import AIDA, LDC, LDC_ONT
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
                    if json_graph.is_type_stmt(stmt) and json_graph.stmt_object(stmt) is not None:
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

#########3
# NEW compute info on clusters needed for handling prototype handles
# returns: ere_set, member_to_clusters, cluster_to_prototype 
def compute_cluster_info(json_graph, ere_list):
    graph_mappings = json_graph.build_cluster_member_mappings()
    member_to_clusters = graph_mappings['member_to_clusters']
    cluster_to_prototype = graph_mappings['cluster_to_prototype']

    return (set(ere_list), member_to_clusters, cluster_to_prototype)


def match_stmt_in_kb(stmt_label, kb_graph, kb_nodes_by_category, json_graph):
    assert json_graph.is_statement(stmt_label)
    stmt_entry = json_graph.node_dict[stmt_label]

    stmt_subj = stmt_entry.subject
    stmt_pred = stmt_entry.predicate
    stmt_obj = stmt_entry.object
    if stmt_subj is None: print("HIER0", stmt_label)
    if stmt_pred is None: print ("HIER1", stmt_label)
    if stmt_obj is None: print ("HIER2", stmt_label)
    assert stmt_subj is not None and stmt_pred is not None and stmt_obj is not None

    # Find the statement node in the KB
    kb_stmt_id = URIRef(stmt_label)
    if kb_stmt_id not in kb_nodes_by_category['Statement']:
        kb_stmt_pred = RDF.type if stmt_pred == 'type' else LDC_ONT.term(stmt_pred)
        kb_stmt_id = next(iter(
                kb_stmt_key_mapping[(URIRef(stmt_subj), kb_stmt_pred, URIRef(stmt_obj))]))

    return kb_stmt_id

#############################
def build_subgraph_for_claim(material_dict, kb_graph, kb_nodes_by_category, json_graph):

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

    # NEW from here
    # type statements
    for stmt in material_dict['type_stmts']:
        kb_stmt_id = match_stmt_in_kb(stmt, kb_graph, kb_nodes_by_category, json_graph)

        if kb_stmt_id is None:
            logging.warning(f"Warning: could not match type statement, skipping: {stmt}")
        else:
            all_triples.update(triples_for_type_stmt(kb_graph, kb_stmt_id))

    # edge statements
    for stmt in material_dict["edge_stmts"]:
        kb_stmt_id = match_stmt_in_kb(stmt, kb_graph, kb_nodes_by_category, json_graph)

        if kb_stmt_id is None:
            logging.warning(f"Warning: could not match edge statement, skipping: {stmt}")
        else:
            all_triples.update(triples_for_edge_stmt(kb_graph, kb_stmt_id))

    # NEW part ends here

    
    #############3
    # logging.info('Constructing a subgraph')
    # Start building the subgraph
    subgraph = Graph()

    # Bind all prefixes of kb_graph to the subgraph
    for prefix, namespace in kb_graph.namespaces():
        if str(namespace) not in [AIDA, LDC, LDC_ONT]:
            subgraph.bind(prefix, namespace)
    # Bind the AIDA, LDC, LDC_ONT, and UTEXAS namespaces to the subgraph
    subgraph.bind('aida', AIDA, override=True)
    subgraph.bind('ldc', LDC, override=True)
    subgraph.bind('ldcOnt', LDC_ONT, override=True)
    subgraph.bind('utexas', UTEXAS)


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


def main():
    parser = ArgumentParser()
    parser.add_argument('claim_id', help="ID of the claim to keep")
    parser.add_argument('filename', help="name of file to use, prefix of both ttl and json filename")
    parser.add_argument('graph_path', help='path to the graph json files')
    parser.add_argument('kb_path', help='path to AIF files')
    parser.add_argument('output_dir', help='path to output directory')
    parser.add_argument('run_id', help='TA3 run ID')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    #####
    # json file: extract associated KEs, and statements linking associated KEs
    # identify json file: should be located directly in graph_path
    json_path = os.path.join(args.graph_path, args.filename +  ".json")
    json_graph = JsonGraph.from_dict(util.read_json_file(json_path, 'JSON graph'))

    # from the json file, extract the correct claim and its associated KEs
    # associated_kes: a list of labels of knowledge elements, both sameas clusters and prototypes
    # associated_stmts: one type statement per KE, plus other statements as long as they connect
    #        two associated KEs
    material_dict = find_claim_associated_kes(args.claim_id, json_graph)
    if material_dict is None:
        print("Error: couldn't find claim", args.claim_id)
        sys.exit(1)

    print("HIER edge stmts", material_dict["edge_stmts"])
    # print("HIER", len(associated_kes), len(associated_stmts))
        
    # for stmt in associated_stmts:
    #     print(stmt)

    ###
    # identify ttl file: can be buried more deeply somewhere under kb_path
    kb_path = find_ttl_file(args.kb_path, args.filename)
    if kb_path is None:
        print("Error: KB not found", args.filename)
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
    rdflib.term._toPythonMapping.pop(rdflib.XSD['gYear'])

    print('Reading kb from {}'.format(kb_path))
    # rdflib graph object
    kb_graph = Graph()
    kb_graph.parse(kb_path, format='ttl')

    kb_nodes_by_category = catalogue_kb_nodes(kb_graph) # NEW

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=not args.force)

    run_id = args.run_id

    subgraph = build_subgraph_for_claim(material_dict, kb_graph, kb_nodes_by_category, json_graph) # NEW

    
    output_path = os.path.join(str(output_dir), run_id + "." + args.filename)
    with open(output_path, 'w') as fout:
        fout.write(print_graph(subgraph))
        


if __name__ == '__main__':
    main()
