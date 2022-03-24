import logging
from collections import defaultdict

from rdflib.namespace import Namespace, RDF, XSD
from rdflib.term import BNode, Literal, URIRef
import rdflib

from aida_utexas.aif import AIF_NS_PREFIX
from aida_utexas.hypothesis.date_check import AidaIncompleteDate

# namespaces for AIDA-related prefixes
AIDA = Namespace(f'{AIF_NS_PREFIX}/InterchangeOntology#')
LDC = Namespace(f'{AIF_NS_PREFIX}/LdcAnnotations#')
LDC_ONT = Namespace(f'{AIF_NS_PREFIX}/LDCOntologyM36#')
EX = Namespace(f'https://www.caci.com/claim-example#')


def count_nodes(node_set):
    """
    Count the number of all nodes and the number of BNodes in the node_set.
    """
    node_count = 0
    bnode_count = 0
    for node_id in node_set:
        node_count += 1
        if not isinstance(node_id, URIRef):
            assert isinstance(node_id, BNode)
            bnode_count += 1
    return node_count, bnode_count


def catalogue_kb_nodes(kb_graph):
    """
    Catalogue all URIRefs and BNodes in the graph by their type (i.e., the object of its type
    statement), and report the count for each type
    :param kb_graph: an object of rdflib.graph.Graph
    :return: a dictionary from each node type to the set of nodes of that specific type
    """
    node_type_set = set()
    # Collect the object of all type statements in the graph, as the set of node types.
    for subj, obj in kb_graph.subject_objects(predicate=RDF.type):
        node_type_set.add(obj)

    # Simplify the node types by removing their prefixes.
    node_type_set = {
        kb_graph.namespace_manager.compute_qname(node_type)[-1] for node_type in node_type_set}
    # Remove the "Statement" type, as we will catalogue type / edge statements separately.
    node_type_set.discard('Statement')

    # The dictionary from each node type to the set of nodes of that specific type
    kb_nodes_by_category = {}

    print('Counts for all node categories in the KB:\n==========')

    # The set of all statement nodes (type statements & edge statements)
    kb_stmt_set = set(kb_graph.subjects(predicate=RDF.type, object=RDF.Statement))
    # The set of all type statement nodes
    kb_type_stmt_set = set(kb_graph.subjects(predicate=RDF.predicate, object=RDF.type))

    assert kb_type_stmt_set.issubset(kb_stmt_set)
    # The set of all edge statement nodes
    kb_edge_stmt_set = kb_stmt_set - kb_type_stmt_set

    kb_nodes_by_category['Statement'] = kb_stmt_set
    kb_nodes_by_category['EdgeStatement'] = kb_edge_stmt_set
    kb_nodes_by_category['TypeStatement'] = kb_type_stmt_set

    for category, node_set in kb_nodes_by_category.items():
        node_count, bnode_count = count_nodes(node_set)
        print('# {}:  {}  (# BNode:  {})'.format(category, node_count, bnode_count))
    print()

    # Count core nodes (EREs, SameAsClusters, and ClusterMemberships)
    for node_type in ['Entity', 'Relation', 'Event', 'SameAsCluster', 'ClusterMembership', 'Claim', 'ClaimComponent']:
        node_set = set(kb_graph.subjects(predicate=RDF.type, object=AIDA.term(node_type)))
        node_count, bnode_count = count_nodes(node_set)
        print('# {}:  {}  (# BNode:  {})'.format(node_type, node_count, bnode_count))
        kb_nodes_by_category[node_type] = node_set
    print()

    # Count justification nodes (TextJustification, ImageJustification, KeyFrameVideoJustification,
    # and CompoundJustification)
    for node_type in sorted(node_type_set):
        if 'Justification' not in node_type:
            continue
        node_set = set(kb_graph.subjects(predicate=RDF.type, object=AIDA.term(node_type)))
        node_count, bnode_count = count_nodes(node_set)
        print('# {}:  {}  (# BNode:  {})'.format(node_type, node_count, bnode_count))
        kb_nodes_by_category[node_type] = node_set
    print()

    # Count other nodes
    for node_type in sorted(node_type_set):
        if node_type in kb_nodes_by_category:
            continue
        node_set = set(kb_graph.subjects(predicate=RDF.type, object=AIDA.term(node_type)))
        node_count, bnode_count = count_nodes(node_set)
        print('# {}:  {}  (# BNode:  {})'.format(node_type, node_count, bnode_count))
        kb_nodes_by_category[node_type] = node_set
    print()

    # Sanity check to make sure that we have all possible node types in the dictionary
    print('Sanity check ...')
    cum_count = 0
    for category, node_set in kb_nodes_by_category.items():
        if category == 'Statement':
            continue
        for node_id in node_set:
            cum_count += len(list(kb_graph.predicate_objects(subject=node_id)))
    if cum_count == len(kb_graph):
        print(
            'Success: the sum of # triples for each node category == '
            '# triples in the whole graph: {}'.format(cum_count))
    else:
        print(
            'Warning: the sum of # triples for each node category != '
            '# triples in the whole graph: {}, {}'.format(cum_count, len(kb_graph)))

    return kb_nodes_by_category


def extract_stmt_components(kb_graph, stmt_id):
    """
    Find the subject, the predicate, and the object of a statement from a graph
    """
    stmt_subj = next(iter(kb_graph.objects(subject=stmt_id, predicate=RDF.subject)))
    stmt_pred = next(iter(kb_graph.objects(subject=stmt_id, predicate=RDF.predicate)))
    stmt_obj = next(iter(kb_graph.objects(subject=stmt_id, predicate=RDF.object)))
    return stmt_subj, stmt_pred, stmt_obj


def index_type_statement_nodes(kb_graph, kb_type_stmt_set=None):
    """
    Build a indexing dictionary from each ERE node to a set of corresponding (type statement node,
    type ontology) pairs for fast lookup, for all type statements in kb_type_stmt_set.
    If kb_type_stmt_set is not provided, then do the indexing for all type statements in kb_graph.
    """
    if kb_type_stmt_set is None:
        kb_type_stmt_set = set(kb_graph.subjects(predicate=RDF.predicate, object=RDF.type))

    kb_type_stmt_key_mapping = defaultdict(set)

    for type_stmt_id in kb_type_stmt_set:
        stmt_subj, stmt_pred, stmt_obj = extract_stmt_components(kb_graph, type_stmt_id)
        assert stmt_pred == RDF.type

        # Get the simplified type string
        type_ont = kb_graph.namespace_manager.compute_qname(stmt_obj)[-1]

        kb_type_stmt_key_mapping[stmt_subj].add((type_stmt_id, type_ont))

    return kb_type_stmt_key_mapping


def index_statement_nodes(kb_graph, kb_stmt_set=None):
    """
    Build an indexing dictionary from each (subject, predicate, object) tuple to a set of
    corresponding statement nodes for fast lookup, for all statements in kb_stmt_set.
    If kb_stmt_set is not provided, then do the indexing for all statements in kb_graph.
    """
    if kb_stmt_set is None:
        kb_stmt_set = set(kb_graph.subjects(predicate=RDF.type, object=RDF.Statement))

    kb_stmt_key_mapping = defaultdict(set)

    for stmt_id in kb_stmt_set:
        stmt_subj, stmt_pred, stmt_obj = extract_stmt_components(kb_graph, stmt_id)
        kb_stmt_key_mapping[(stmt_subj, stmt_pred, stmt_obj)].add(stmt_id)

    return kb_stmt_key_mapping


def index_cluster_membership_nodes(kb_graph, kb_cm_set=None):
    """
    Build an indexing dictionary from each (cluster, member) pairs to a set of corresponding
    ClusterMembership nodes for fast lookup, for all ClusterMembership nodes in kb_cm_set.
    If kb_cm_set is not provided, then do the indexing for all ClusterMembership nodes in kb_graph.
    """
    if kb_cm_set is None:
        kb_cm_set = set(kb_graph.subjects(predicate=RDF.type, object=AIDA.ClusterMembership))

    kb_cm_key_mapping = defaultdict(set)

    for kb_cm in kb_cm_set:
        kb_cm_cluster = next(iter(kb_graph.objects(subject=kb_cm, predicate=AIDA.cluster)))
        kb_cm_member = next(iter(kb_graph.objects(subject=kb_cm, predicate=AIDA.clusterMember)))
        kb_cm_key_mapping[(kb_cm_cluster, kb_cm_member)].add(kb_cm)

    return kb_cm_key_mapping


def match_subjects_intersection(kb_graph, po_pair_list, verbose=False):
    """
    Find the first subject node (if any) that matches all (predicate, object) constraints as
    specified in po_pair_list, or None if no match can be found. When verbose is True, print
    warning messages if there are no match or more than 1 matches.
    This is a helper function for match_statement_bnode and match_cluster_membership_bnode.
    """
    po_pair = po_pair_list[0]
    match_set = set(kb_graph.subjects(predicate=po_pair[0], object=po_pair[1]))

    for po_pair in po_pair_list[1:]:
        match_set = match_set & set(kb_graph.subjects(predicate=po_pair[0], object=po_pair[1]))

    warning_msg = ' & '.join([str(po_pair[0]) for po_pair in po_pair_list])

    if len(match_set) == 0:
        if verbose:
            print('Warning: cannot find a match for {}!'.format(warning_msg))
        return None

    if verbose and len(match_set) > 1:
        print('Warning: find more than 1 match for {}!'.format(warning_msg))

    return match_set.pop()


def match_statement_bnode(kb_graph, stmt_entry, kb_nodes_by_category=None, verbose=False):
    """
    Find the statement node from the kb_graph based on the subject/predicate/object information
    in the stmt_entry from the json graph. If kb_nodes_by_category is provided, will also verify
    if the matched node is in the set of TypeStatement/EdgeStatement.
    This is not used anymore due to index_statement_nodes.
    """
    assert stmt_entry['type'] == 'Statement'
    stmt_subj = stmt_entry.get('subject', None)
    stmt_pred = stmt_entry.get('predicate', None)
    stmt_obj = stmt_entry.get('object', None)
    assert stmt_subj is not None and stmt_pred is not None and stmt_obj is not None

    kb_stmt_id = match_subjects_intersection(
        kb_graph=kb_graph,
        po_pair_list=[(RDF.subject, URIRef(stmt_subj)), (RDF.object, URIRef(stmt_obj))],
        verbose=verbose)

    if kb_stmt_id is not None and kb_nodes_by_category is not None:
        if stmt_pred == 'type':
            assert kb_stmt_id in kb_nodes_by_category['TypeStatement']
        else:
            assert kb_stmt_id in kb_nodes_by_category['EdgeStatement']

    return kb_stmt_id


def match_cluster_membership_bnode(kb_graph, cm_entry, kb_nodes_by_category=None, verbose=False):
    """
    Find the cluster membership node from the kb_graph based on the cluster/clusterMember
    information in the cm_entry from the json graph. If kb_nodes_by_category is provided, will also
    verify if the matched node is in the set of ClusterMembership.
    This is not used anymore due to index_cluster_membership_nodes.
    """
    assert cm_entry['type'] == 'ClusterMembership'
    cm_cluster = cm_entry['cluster']
    cm_member = cm_entry['clusterMember']

    kb_cm_id = match_subjects_intersection(
        kb_graph=kb_graph,
        po_pair_list=[(AIDA.cluster, URIRef(cm_cluster)), (AIDA.clusterMember, URIRef(cm_member))],
        verbose=verbose)

    if kb_cm_id is not None and kb_nodes_by_category is not None:
        assert kb_cm_id in kb_nodes_by_category['ClusterMembership']

    return kb_cm_id

###
# KE Feb 2021: catching errors in the triple extraction, namely triples that contain None:
# warn, and skip faulty triples
def update_triples_catchnone(triples, added_triples, info):
    for t in added_triples:
        if any(component is None for component in t):
            logging.warning("Error invalid KB triple created: {}\n{}\n{}\n{}".format(info, t[0], t[1], t[2]))
            

    triples.update([t for t in added_triples if not(any(component is None for component in t))])

def triples_for_subject(kb_graph, query_subj, expanding_preds=None, excluding_preds=None):
    """
    Extract all triples from kb_graph with query_subj as the subject. If expanding_preds is
    provided, all objects of predicates in the expanding_preds will be recursively expanded using
    the same function. If excluding_preds is provided, triples who predicates are in the
    excluding_preds will be ignored.
    """
    if expanding_preds is None:
        expanding_preds = []
    if excluding_preds is None:
        excluding_preds = []

    triples = set()

    for s, p, o in kb_graph.triples((query_subj, None, None)):
        if p in excluding_preds:
            continue
        update_triples_catchnone(triples, [(s, p, o)], "triples for subject")
        if p in expanding_preds:
            update_triples_catchnone(triples, triples_for_subject(
                kb_graph, o, expanding_preds=expanding_preds, excluding_preds=excluding_preds), "triples for subject loc. 2")

    return triples


def triples_for_edge_stmt(kb_graph, stmt_id):
    """
    Extracting all triples related to a edge statement node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((stmt_id, None, None)):
        if p == AIDA.privateData:
            continue

        if p == AIDA.justifiedBy:
            update_triples_catchnone(triples, [(s, p, o)], "triples for edge stmt justification")
            update_triples_catchnone(triples, triples_for_compound_just(kb_graph, o), "triples for edge stmt compound justification")
        else:
            update_triples_catchnone(triples, [(s, p, o)], "triples for edge stmt")

            update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "triples for edge stmt conf and system")

    return triples


def triples_for_type_stmt(kb_graph, stmt_id):
    """
    Extracting all triples related to a type statement node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((stmt_id, None, None)):
        if p == AIDA.privateData:
            continue

        # basically triples.add((s, p, o))
        update_triples_catchnone(triples, [(s, p, o)], "triples for type stmt")

        if p == AIDA.justifiedBy:
            # basically triples.update(triples_for_justification(kb_graph, o))
            update_triples_catchnone(triples, triples_for_justification(kb_graph, o), "triples for type stmt: justification")

        update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "triples for type stmt: conf and system")

    return triples


def triples_for_ere(kb_graph, ere_id):
    """
    Extracting all triples related to an ERE node. Will also check whether there are informative
    justifications sharing the same source document id. If so, will apply a patch.
    """
    triples = set()

    # Hot fix to ensure that each ERE has up to one informative justification per source document
    seen_source_document = set()

    for s, p, info_just in kb_graph.triples((ere_id, AIDA.informativeJustification, None)):
        info_just_triples = triples_for_justification(kb_graph, info_just)
        
        source_document = None
        for _, info_just_p, info_just_o in info_just_triples:
            if info_just_p == AIDA.sourceDocument:
                source_document = str(info_just_o)
                break

        if source_document is not None:
            if source_document in seen_source_document:
                logging.warning(
                    'Duplicate source document {} in informative justifications for {}'.format(
                        source_document, ere_id))
                continue
            seen_source_document.add(source_document)

        update_triples_catchnone(triples, info_just_triples, "for ere: justifications")
        update_triples_catchnone(triples, [(s, p, info_just)], "for ere: informative just")

    for s, p, o in kb_graph.triples((ere_id, None, None)):
        if p in [AIDA.justifiedBy, AIDA.privateData]:
            continue

        if p == AIDA.informativeJustification:
            continue

        update_triples_catchnone(triples, [(s, p, o)], "triples for ere: graph triple")
        
        if p == AIDA.ldcTime:            
            update_triples_catchnone(triples, triples_for_ldc_time(kb_graph, o), "for ere: ldc time")
        if p == AIDA.link:
            update_triples_catchnone(triples, triples_for_link_assertion(kb_graph, o), "for ere: link assertion")

        update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "for ere: conf and system node")

    return triples


def triples_for_cluster(kb_graph, cluster_id):
    """
    Extracting all triples related to a SameAsCluster node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((cluster_id, None, None)):
        if p in [AIDA.justifiedBy, AIDA.privateData, AIDA.link, AIDA.ldcTime]:
            continue

        update_triples_catchnone(triples, [(s, p, o)], "triples for cluster")

        if p == AIDA.informativeJustification:
            update_triples_catchnone(triples, triples_for_justification(kb_graph, o), "triples for cluster: justification")

        update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "triples for cluster: conf and system")

    return triples


def triples_for_cluster_membership(kb_graph, cm_id):
    """
    Extracting all triples related to a ClusterMembership node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((cm_id, None, None)):
        update_triples_catchnone(triples, [(s, p, o)], "triples for cluster membership")

        update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "triples for cluster membership: conf and system")

    return triples


def triples_for_claim(kb_graph, claim_id):
    """
    Extracting all triples related to a claim node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((claim_id, None, None)):
        if p == AIDA.privateData:
            continue

        if p == AIDA.justifiedBy:
            continue
            #hot fix: justification node for claim lack source document
            #update_triples_catchnone(triples, [(s, p, o)], "triples for claim: justification")
            #update_triples_catchnone(triples, triples_for_compound_just(kb_graph, o), "triples for claim: compound justification")
        else:
            update_triples_catchnone(triples, [(s, p, o)], "triples for claim")

            update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "triples for claim: conf and system")
        
        if p == AIDA.claimDateTime: 
            update_triples_catchnone(triples, triples_for_ldc_time(kb_graph, o), "for claim: ldc time")
        if p == AIDA.link:
            update_triples_catchnone(triples, triples_for_link_assertion(kb_graph, o), "for claim: link assertion")

        if p == AIDA.claimLocation or p == AIDA.claimer or p == AIDA.xVariable or p == AIDA.claimerAffiliation:
            update_triples_catchnone(triples, triples_for_claimcomponent(kb_graph, o), "for claim: claim component")
    
    if len(list(kb_graph.objects(subject=claim_id, predicate=AIDA.sentiment))) == 0:
        update_triples_catchnone(triples, [(claim_id, AIDA.sentiment, AIDA.SentimentNeutralUnknown)],"for claim: claim sentiment manually adding" )


    #if len(list(kb_graph.objects(subject=claim_id, predicate=AIDA.sentiment))) == 0:
        #update_triples_catchnone(triples, [(claim_id, AIDA.sentiment, AIDA.SentimentNeutralUnknown)], "sentiment for claim")

    return triples

def triples_for_claimcomponent(kb_graph, comp_id):
    """
    Extracting all triples related to a claim coponent node.
    """
    triples = set()
    countType = 0
    
    for s, p, o in kb_graph.triples((comp_id, None, None)):
        if p == AIDA.privateData:
            continue
        
        #jy: hot fix for validator error
        if p == AIDA.componentKE:
            continue

        if p == AIDA.componentType: 
            countType = countType + 1
            if countType <= 5:
                update_triples_catchnone(triples, [(s, p, o)], "triples for claim component: componentType")
            else:
                continue

        if p == AIDA.justifiedBy:
            update_triples_catchnone(triples, [(s, p, o)], "triples for claim component: justification")
            update_triples_catchnone(triples, triples_for_compound_just(kb_graph, o), "triples for claim component: compound justification")
        else:
            update_triples_catchnone(triples, [(s, p, o)], "triples for claim")

            update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "triples for claim component: conf and system")

        if p == AIDA.link:
            update_triples_catchnone(triples, triples_for_link_assertion(kb_graph, o), "for claim component: link assertion")

    return triples

def triples_for_compound_just(kb_graph, comp_just_id):
    """
    Extracting all triples related to a CompoundJustification node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((comp_just_id, None, None)):
        if p == AIDA.privateData:
            continue

        update_triples_catchnone(triples, [(s, p, o)], "triples for compound just")

        if p == AIDA.containedJustification:
            update_triples_catchnone(triples, triples_for_justification(kb_graph, o), "triples for compound just: justification")

        update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "triples for compound just: conf and system")

    return triples


def triples_for_justification(kb_graph, just_id):
    """
    Extracting all triples related to a Justification node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((just_id, None, None)):
        if p == AIDA.privateData:
            continue

        update_triples_catchnone(triples, [(s, p, o)], "triples for justification")

        # The aida:boundingBox field might also contain aida:system, so need to expand.
        if p == AIDA.boundingBox:
            update_triples_catchnone(triples, triples_for_subject(kb_graph, o, expanding_preds=[AIDA.system]), "triples for just: expanded")

        update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "triples for just: conf and system")

    return triples


def triples_for_conf(kb_graph, conf_id):
    """
    Extracting all triples related to a Confidence node. Will also check if the confidence value
    is too small (< 1e-4). If so, will apply a patch.
    """
    triples = set()

    for s, p, o in kb_graph.triples((conf_id, None, None)):
        # Hot fix to ensure that the confidence value is not too small.
        ### jy
        # alter how to present confidence Value
        if p == AIDA.confidenceValue:
            '''
            if float(o) < 0.0001:
                conf_value = Literal(0.0001, datatype=XSD.double)
            else:
                conf_value = o
            '''
            conf_value = o

            update_triples_catchnone(triples, [(s, p, conf_value)], "triples for conf")

        else:
            update_triples_catchnone(triples, [(s, p, o)], "triples for conf loc 2")

        if p == AIDA.system:
            update_triples_catchnone(triples, triples_for_subject(kb_graph, o), "triples for conf: system")

    return triples


def triples_for_link_assertion(kb_graph, link_assertion_id):
    """
    Extracting all triples related to an LinkAssertion node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((link_assertion_id, None, None)):
        update_triples_catchnone(triples, [(s, p, o)], "triples for link assertion")

        update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "triples for link assertion: conf and system")

    return triples

def triples_for_ldc_time(kb_graph, time_id):
    """
    Extracting all triples related to an LDCTime node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((time_id, None, None)):
        update_triples_catchnone(triples, [(s, p, o)], "for ldc time: graph triples")
        if p in [AIDA.start, AIDA.end]:
            update_triples_catchnone(triples, triples_for_time(kb_graph, s, p), "for ldc time; aidastart aidaend")
        update_triples_catchnone(triples, expand_conf_and_system_node(kb_graph, p, o), "for ldc time: conf and system")

    return triples

def triples_for_time(kb_graph, time_id, p):
    """
    Extracting all triples containing LDCTimeComponent nodes related to an LDCTime node.
    """
    triples = set()

    before_time_component_id, after_time_component_id = None, None
    for _, _, time_component_id in kb_graph.triples((time_id, p, None)):
        for s, p, o in kb_graph.triples((time_component_id, None, None)):
            if p in [AIDA.year, AIDA.month, AIDA.day]:
                continue
            update_triples_catchnone(triples, [(s, p, o)], "for time")         
            if p == AIDA.timeType and o == Literal("BEFORE"):# jy correct the datatype of o
                before_time_component_id = s
            elif p == AIDA.timeType and o == Literal("AFTER"):# jy correct the datatype of o
                after_time_component_id = s
                
    before_year, before_month, before_day = None, None, None
    if before_time_component_id:
        for s, p, o in kb_graph.triples((before_time_component_id, AIDA.year, None)):
            #jy : there is a bug that for gYear 2020 it will be parsed as 2020-01-01
            #therefore we should remove the month and day, and only maintain the year 02/24/2022
            #we first treat the year as an intege to avoid it becomes a year-month-day format in the middle of the process
            #we will transform the year into xsd.gYear at end when adding it to tuples
            year_parts = str(o).split('-')          
            before_year =  Literal(year_parts[0], datatype=XSD.int)
            
        for s, p, o in kb_graph.triples((before_time_component_id, AIDA.month, None)):
            before_month = o
                       
        for s, p, o in kb_graph.triples((before_time_component_id, AIDA.day, None)):
            before_day = o

    after_year, after_month, after_day = None, None, None
    if after_time_component_id:
        for s, p, o in kb_graph.triples((after_time_component_id, AIDA.year, None)):
            #jy : there is a bug that for gYear 2020 it will be parsed as 2020-01-01
            #therefore we should remove the month and day, and only maintain the year 02/24/2022
            #we first treat the year as an intege to avoid it becomes a year-month-day format in the middle of the process
            #we will transform the year into xsd.gYear at end when adding it to tuples
            year_parts = str(o).split('-')
            after_year = Literal(year_parts[0], datatype=XSD.int)

        for s, p, o in kb_graph.triples((after_time_component_id, AIDA.month, None)):
            after_month = o

        for s, p, o in kb_graph.triples((after_time_component_id, AIDA.day, None)):
            after_day = o
    
    before_year_value = int(''.join(filter(lambda i: i.isdigit(), str(before_year)))) if before_year else None
    before_month_value = int(''.join(filter(lambda i: i.isdigit(), str(before_month)))) if before_month else None
    before_day_value = int(''.join(filter(lambda i: i.isdigit(), str(before_day)))) if before_day else None
    after_year_value = int(''.join(filter(lambda i: i.isdigit(), str(after_year)))) if after_year else None
    after_month_value = int(''.join(filter(lambda i: i.isdigit(), str(after_month)))) if after_month else None
    after_day_value = int(''.join(filter(lambda i: i.isdigit(), str(after_day)))) if after_day else None
     
    # after_date should be no later than before_date
    before_date = AidaIncompleteDate(before_year_value, before_month_value, before_day_value)
    after_date = AidaIncompleteDate(after_year_value, after_month_value, after_day_value)
    
    # jy
    # There are before date is before after date in claim files so ignore this date correction
    
    # if before_date.is_before(after_date):
    #    before_year = after_year
    #    before_month = after_month
    #    before_day = after_day
      
    if before_year:
        update_triples_catchnone(triples, [(before_time_component_id, AIDA.year, Literal(before_year, datatype = XSD.gYear, normalize=False))], "for time: before year") #jy: make sure year is xsd.gYear type
    if before_month:
        update_triples_catchnone(triples, [(before_time_component_id, AIDA.month, before_month)], "for time: before month")
    if before_day:
        update_triples_catchnone(triples, [(before_time_component_id, AIDA.day, before_day)], "for time: before day")
    if after_year:
        update_triples_catchnone(triples, [(after_time_component_id, AIDA.year, Literal(after_year, datatype = XSD.gYear, normalize=False))], 'for time: after year') #jy: make sure year is xsd.gYear type
    if after_month:
        update_triples_catchnone(triples, [(after_time_component_id, AIDA.month, after_month)], "for time: after month")
    if after_day:
        update_triples_catchnone(triples, [(after_time_component_id, AIDA.day, after_day)], "for time: after day")
    
    return triples

def expand_conf_and_system_node(kb_graph, p, o):
    """
    Expand the triples when the predicate is aida:confidence or aida:system. This is a helper
    function to reduce code duplications in the above functions.
    """
    triples = set()

    if p == AIDA.confidence:
        update_triples_catchnone(triples, triples_for_conf(kb_graph, o), "triples for conf and system: conf")

    if p == AIDA.system:
        update_triples_catchnone(triples, triples_for_subject(kb_graph, o), "triples for conf and system: system")

    return triples
