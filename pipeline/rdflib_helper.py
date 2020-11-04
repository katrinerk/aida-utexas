import logging
from collections import defaultdict

from rdflib.namespace import Namespace, RDF, XSD
from rdflib.term import BNode, Literal, URIRef

from aida_utexas.aif import AIF_NS_PREFIX

# namespaces for AIDA-related prefixes
AIDA = Namespace(f'{AIF_NS_PREFIX}/InterchangeOntology#')
LDC = Namespace(f'{AIF_NS_PREFIX}/LdcAnnotations#')
LDC_ONT = Namespace(f'{AIF_NS_PREFIX}/LDCOntologyM36#')


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
    for node_type in ['Entity', 'Relation', 'Event', 'SameAsCluster', 'ClusterMembership']:
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
        triples.add((s, p, o))
        if p in expanding_preds:
            triples.update(triples_for_subject(
                kb_graph, o, expanding_preds=expanding_preds, excluding_preds=excluding_preds))

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
            triples.add((s, p, o))
            triples.update(triples_for_compound_just(kb_graph, o))
        else:
            triples.add((s, p, o))

            triples.update(expand_conf_and_system_node(kb_graph, p, o))

    return triples


def triples_for_type_stmt(kb_graph, stmt_id):
    """
    Extracting all triples related to a type statement node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((stmt_id, None, None)):
        if p == AIDA.privateData:
            continue

        triples.add((s, p, o))

        if p == AIDA.justifiedBy:
            triples.update(triples_for_justification(kb_graph, o))

        triples.update(expand_conf_and_system_node(kb_graph, p, o))

    return triples


def triples_for_ere(kb_graph, ere_id):
    """
    Extracting all triples related to an ERR node. Will also check whether there are informative
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

        triples.add((s, p, info_just))
        triples.update(info_just_triples)

    for s, p, o in kb_graph.triples((ere_id, None, None)):
        if p in [AIDA.justifiedBy, AIDA.privateData]:
            continue

        if p == AIDA.informativeJustification:
            continue

        triples.add((s, p, o))

        if p == AIDA.ldcTime:
            triples.update(triples_for_ldc_time(kb_graph, o))
        if p == AIDA.link:
            triples.update(triples_for_link_assertion(kb_graph, o))

        triples.update(expand_conf_and_system_node(kb_graph, p, o))

    return triples


def triples_for_cluster(kb_graph, cluster_id):
    """
    Extracting all triples related to a SameAsCluster node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((cluster_id, None, None)):
        if p in [AIDA.justifiedBy, AIDA.privateData, AIDA.link, AIDA.ldcTime]:
            continue

        triples.add((s, p, o))

        if p == AIDA.informativeJustification:
            triples.update(triples_for_justification(kb_graph, o))

        triples.update(expand_conf_and_system_node(kb_graph, p, o))

    return triples


def triples_for_cluster_membership(kb_graph, cm_id):
    """
    Extracting all triples related to a ClusterMembership node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((cm_id, None, None)):
        triples.add((s, p, o))

        triples.update(expand_conf_and_system_node(kb_graph, p, o))

    return triples


def triples_for_compound_just(kb_graph, comp_just_id):
    """
    Extracting all triples related to a CompoundJustification node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((comp_just_id, None, None)):
        if p == AIDA.privateData:
            continue

        triples.add((s, p, o))

        if p == AIDA.containedJustification:
            triples.update(triples_for_justification(kb_graph, o))

        triples.update(expand_conf_and_system_node(kb_graph, p, o))

    return triples


def triples_for_justification(kb_graph, just_id):
    """
    Extracting all triples related to a Justification node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((just_id, None, None)):
        if p == AIDA.privateData:
            continue

        triples.add((s, p, o))

        # The aida:boundingBox field might also contain aida:system, so need to expand.
        if p == AIDA.boundingBox:
            triples.update(triples_for_subject(kb_graph, o, expanding_preds=[AIDA.system]))

        triples.update(expand_conf_and_system_node(kb_graph, p, o))

    return triples


def triples_for_conf(kb_graph, conf_id):
    """
    Extracting all triples related to a Confidence node. Will also check if the confidence value
    is too small (< 1e-4). If so, will apply a patch.
    """
    triples = set()

    for s, p, o in kb_graph.triples((conf_id, None, None)):
        # Hot fix to ensure that the confidence value is not too small.
        if p == AIDA.confidenceValue:
            if float(o) < 0.0001:
                conf_value = Literal(0.0001, datatype=XSD.double)
            else:
                conf_value = o

            triples.add((s, p, conf_value))

        else:
            triples.add((s, p, o))

        if p == AIDA.system:
            triples.update(triples_for_subject(kb_graph, o))

    return triples


def triples_for_link_assertion(kb_graph, link_assertion_id):
    """
    Extracting all triples related to an LinkAssertion node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((link_assertion_id, None, None)):
        triples.add((s, p, o))

        triples.update(expand_conf_and_system_node(kb_graph, p, o))

    return triples


def triples_for_ldc_time(kb_graph, time_id):
    """
    Extracting all triples related to an LDCTime node.
    """
    triples = set()

    for s, p, o in kb_graph.triples((time_id, None, None)):
        triples.add((s, p, o))
        if p in [AIDA.start, AIDA.end]:
            triples.update(triples_for_subject(kb_graph, o))

        triples.update(expand_conf_and_system_node(kb_graph, p, o))

    return triples


def expand_conf_and_system_node(kb_graph, p, o):
    """
    Expand the triples when the predicate is aida:confidence or aida:system. This is a helper
    function to reduce code duplications in the above functions.
    """
    triples = set()

    if p == AIDA.confidence:
        triples.update(triples_for_conf(kb_graph, o))

    if p == AIDA.system:
        triples.update(triples_for_subject(kb_graph, o))

    return triples
