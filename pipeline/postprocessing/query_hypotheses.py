import json
from argparse import ArgumentParser
from operator import itemgetter

from aida_utexas import sparql_helper
from aida_utexas import util
from aida_utexas.aif import JsonGraph
from aida_utexas.json_graph_helper import build_cluster_member_mappings

AIF_HEADER_PREFIXES = \
    '@prefix ldcOnt: <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LDCOntology#> .\n' \
    '@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n' \
    '@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .\n' \
    '@prefix aida:  <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/InterchangeOntology#> .\n' \
    '@prefix ldc:   <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LdcAnnotations#> .\n'

QUERY_PREFIXES = \
    'PREFIX ldcOnt: <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LDCOntology#>\n' \
    'PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n' \
    'PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>\n' \
    'PREFIX aida:  <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/InterchangeOntology#>\n' \
    'PREFIX ldc:   <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LdcAnnotations#>\n'


def queries_for_aida_result(
        json_graph, hypothesis, member_to_clusters, cluster_to_prototype, prototype_set,
        num_node_queries=5, num_stmts_per_query=3000,
        query_just=False, query_conf=False):
    ere_id_list = []

    ere_query_item_list = []
    cluster_query_item_list = []
    stmt_query_item_list = []
    just_query_item_list = []
    conf_query_item_list = []

    # Extract all aida:system definitions.
    stmt_query_item_list.append('{?x a aida:System .}')

    for stmt_label in hypothesis['statements']:
        assert json_graph.is_statement(stmt_label)

        subject_id = json_graph.stmt_subject(stmt_label)
        predicate_id = json_graph.stmt_predicate(stmt_label)
        object_id = json_graph.stmt_object(stmt_label)

        # add subject to node_str_list for node query
        ere_query_item_list.append('<{}>'.format(subject_id))
        ere_id_list.append(subject_id)

        if predicate_id != 'type':
            # add object to node_str_list for node query if
            # it's not a typing statement
            ere_query_item_list.append('<{}>'.format(object_id))
            ere_id_list.append(object_id)

            stmt_constraint = \
                '?x a rdf:Statement .\n?x rdf:subject <{}> .\n?x rdf:predicate ldcOnt:{} .\n' \
                '?x rdf:object <{}> .'.format(
                    subject_id, predicate_id, object_id)

            stmt_query_item_list.append('{{\n{}\n}}'.format(stmt_constraint))

            if query_just:
                just_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:justifiedBy ?j .\n'
                    '}}'.format(stmt_constraint))
                just_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:justifiedBy ?cj .\n'
                    '?cj aida:containedJustification ?j .\n'
                    '}}'.format(stmt_constraint))

            if query_conf:
                conf_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:confidence ?c .\n'
                    '}}'.format(stmt_constraint))

                conf_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:justifiedBy ?j .\n'
                    '?j aida:confidence ?c .\n'
                    '}}'.format(stmt_constraint))
                conf_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:justifiedBy ?cj .\n'
                    '?cj aida:containedJustification ?j .\n'
                    '?j aida:confidence ?c .\n'
                    '}}'.format(stmt_constraint))

            # just_query_item_list.append(
            #     '{{\n?x a rdf:Statement .\n'
            #     '?x rdf:subject <{}> .\n'
            #     '?x rdf:predicate ldcOnt:{} .\n'
            #     '?x rdf:object <{}> .\n'
            #     '?x aida:justifiedBy ?cj. FILTER isIRI(?cj)\n'
            #     'OPTIONAL {{ ?cj aida:confidence ?cc. FILTER isIRI(?cc) }}\n'
            #     'OPTIONAL {{\n'
            #     # '?cj a aida:CompoundJustification.\n'
            #     '?cj aida:containedJustification ?j. FILTER isIRI(?j) \n'
            #     'OPTIONAL {{ ?j aida:confidence ?c. FILTER isIRI(?cc) }}\n'
            #     '}}\n}}'.format(
            #         subject_id, predicate_id, object_id))

        else:
            # Exclude typing statements of non-prototype members to reduce file size
            # if subject_id not in prototype_set:
            #     continue

            stmt_constraint = \
                '?x a rdf:Statement .\n?x rdf:subject <{}> .\n?x rdf:predicate rdf:type .\n' \
                '?x rdf:object <{}> .'.format(subject_id, object_id)

            stmt_query_item_list.append('{{\n{}\n}}'.format(stmt_constraint))

            if query_just:
                just_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:justifiedBy ?j .\n'
                    '}}'.format(stmt_constraint))

            if query_conf:
                conf_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:confidence ?c .\n'
                    '}}'.format(stmt_constraint))

                conf_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:justifiedBy ?j .\n'
                    '?j aida:confidence ?c .\n'
                    '}}'.format(stmt_constraint))

            # just_query_item_list.append(
            #     '{{\n?x a rdf:Statement .\n'
            #     '?x rdf:subject <{}> .\n'
            #     '?x rdf:predicate rdf:type .\n'
            #     '?x rdf:object <{}> .\n'
            #     '?x aida:justifiedBy ?j. FILTER isIRI(?j)\n'
            #     'OPTIONAL {{ ?j aida:confidence ?c . FILTER isIRI(?c) }}\n}}'.format(
            #         subject_id, object_id))

    for ere_id in ere_id_list:
        # just_query_item_list.append(
        #     '{{\n<{}> aida:justifiedBy ?j. FILTER isIRI(?j)\n'
        #     'OPTIONAL {{ ?j aida:confidence ?c . FILTER isIRI(?c) }}\n}}'.format(ere_id))
        # just_query_item_list.append(
        #     '{{\n<{}> aida:informativeJustification ?j. FILTER isIRI(?j)\n'
        #     'OPTIONAL {{ ?j aida:confidence ?c . FILTER isIRI(?c) }}\n}}'.format(ere_id))

        if query_just:
            just_query_item_list.append(
                '{{<{}> aida:informativeJustification ?j .}}'.format(ere_id))
        if query_conf:
            conf_query_item_list.append(
                '{{\n'
                '<{}> aida:informativeJustification ?j .\n'
                '?j aida:confidence ?c .\n'
                '}}'.format(ere_id))

        for cluster_id in member_to_clusters[ere_id]:
            cluster_query_item_list.append('<{}>'.format(cluster_id))

            stmt_query_item_list.append(
                '{{?x a aida:ClusterMembership .\n'
                '?x aida:cluster <{}> .\n'
                '?x aida:clusterMember <{}> .\n'
                '}}'.format(cluster_id, ere_id))

            # just_query_item_list.append(
            #     '{{\n<{}> aida:informativeJustification ?j. FILTER isIRI(?j)\n'
            #     'OPTIONAL {{ ?j aida:confidence ?c . FILTER isIRI(?c) }}\n}}'.format(cluster_id))

            if query_just:
                just_query_item_list.append(
                    '{{<{}> aida:informativeJustification ?j .}}'.format(cluster_id))

            if query_conf:
                conf_query_item_list.append(
                    '{{\n'
                    '<{}> aida:informativeJustification ?j .\n'
                    '?j aida:confidence ?c .\n'
                    '}}'.format(cluster_id))

                conf_query_item_list.append(
                    '{{?x a aida:ClusterMembership .\n'
                    '?x aida:cluster <{}> .\n'
                    '?x aida:clusterMember <{}> .\n'
                    '?x aida:confidence ?c .\n'
                    '}}'.format(cluster_id, ere_id))

            # Always add the prototype member of the clusters included in the hypothesis.
            prototype_id = cluster_to_prototype[cluster_id]
            ere_query_item_list.append('<{}>'.format(prototype_id))

            # Add the informative justification of the prototype if needed.
            if query_just:
                just_query_item_list.append(
                    '{{<{}> aida:informativeJustification ?j .}}'.format(prototype_id))
            # Add the confidence node of the informative justification of the prototype if needed
            if query_conf:
                conf_query_item_list.append(
                    '{{\n'
                    '<{}> aida:informativeJustification ?j .\n'
                    '?j aida:confidence ?c .\n'
                    '}}'.format(prototype_id))

            # Also add the ClusterMembership nodes for the prototype.
            for proto_cluster_id in member_to_clusters[prototype_id]:
                stmt_query_item_list.append(
                    '{{?x a aida:ClusterMembership .\n'
                    '?x aida:cluster <{}> .\n'
                    '?x aida:clusterMember <{}> .\n'
                    '}}'.format(proto_cluster_id, prototype_id))

                # And the confidence node of the ClusterMembership node if needed
                if query_conf:
                    conf_query_item_list.append(
                        '{{?x a aida:ClusterMembership .\n'
                        '?x aida:cluster <{}> .\n'
                        '?x aida:clusterMember <{}> .\n'
                        '?x aida:confidence ?c .\n'
                        '}}'.format(proto_cluster_id, prototype_id))

            # Also add the typing statement for the prototype.
            proto_stmt_constraint = \
                '?x a rdf:Statement .\n?x rdf:subject <{}> .\n?x rdf:predicate rdf:type .'.format(
                    prototype_id)

            stmt_query_item_list.append('{{\n{}\n}}'.format(proto_stmt_constraint))

            if query_just:
                just_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:justifiedBy ?j .\n'
                    '}}'.format(proto_stmt_constraint))

            if query_conf:
                conf_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:confidence ?c .\n'
                    '}}'.format(proto_stmt_constraint))

                conf_query_item_list.append(
                    '{{\n{}\n'
                    '?x aida:justifiedBy ?j .\n'
                    '?j aida:confidence ?c .\n'
                    '}}'.format(proto_stmt_constraint))

    node_query_item_list = list(set(ere_query_item_list)) + list(set(cluster_query_item_list))
    stmt_query_item_list = list(set(stmt_query_item_list))
    just_query_item_list = list(set(just_query_item_list))

    node_query_list = sparql_helper.produce_node_queries(
        node_query_item_list, num_node_queries=num_node_queries)

    stmt_query_list = sparql_helper.produce_stmt_queries(
        stmt_query_item_list, query_prefixes=QUERY_PREFIXES,
        num_stmts_per_query=num_stmts_per_query)

    just_query_list = sparql_helper.produce_just_queries(
        just_query_item_list, query_prefixes=QUERY_PREFIXES,
        num_stmts_per_query=num_stmts_per_query)

    conf_query_list = sparql_helper.produce_conf_queries(
        conf_query_item_list, query_prefixes=QUERY_PREFIXES,
        num_stmts_per_query=num_stmts_per_query)

    return node_query_list, stmt_query_list, just_query_list, conf_query_list


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_json_path', help='path to the graph json file')
    parser.add_argument('hypotheses_json_path', help='path to the hypotheses json file')
    parser.add_argument('db_path_prefix', help='prefix of tdb database path')
    parser.add_argument('output_dir', help='path to output directory')
    parser.add_argument('--top', default=14, type=int,
                        help='number of top hypothesis to output')
    parser.add_argument('--dry_run', action='store_true',
                        help='if specified, only write the SPARQL queries to '
                             'files, without actually executing the queries')
    parser.add_argument('--query_just', action='store_true')
    parser.add_argument('--query_conf', action='store_true')

    args = parser.parse_args()

    graph_json_path = util.get_input_path(args.graph_json_path)
    print('Reading the graph from {}'.format(graph_json_path))
    with open(str(graph_json_path), 'r') as fin:
        json_graph = JsonGraph.from_dict(json.load(fin))

    mappings = build_cluster_member_mappings(json_graph)
    member_to_clusters = mappings['member_to_clusters']
    cluster_to_prototype = mappings['cluster_to_prototype']
    prototype_set = set(mappings['prototype_to_clusters'].keys())

    hypotheses_json_path = util.get_input_path(args.hypotheses_json_path)
    print('Reading the hypotheses from {}'.format(hypotheses_json_path))
    with open(str(hypotheses_json_path), 'r') as fin:
        hypotheses_json = json.load(fin)

    print('Found {} hypotheses with probabilities of {}'.format(
        len(hypotheses_json['probs']), hypotheses_json['probs']))

    output_dir = util.get_dir(args.output_dir, create=True)

    db_path_prefix = util.get_dir(args.db_path_prefix)
    db_path_list = [str(path) for path in sorted(db_path_prefix.glob('copy*'))]
    print('Using the following tdb databases to query: {}'.format(db_path_list))

    num_node_queries = len(db_path_list)

    top_count = 0
    for result_idx, prob in sorted(
            enumerate(hypotheses_json['probs']), key=itemgetter(1), reverse=True):
        hypothesis = hypotheses_json['support'][result_idx]
        node_query_list, stmt_query_list, just_query_list, conf_query_list = \
            queries_for_aida_result(
                json_graph=json_graph,
                hypothesis=hypothesis,
                member_to_clusters=member_to_clusters,
                cluster_to_prototype=cluster_to_prototype,
                prototype_set=prototype_set,
                num_node_queries=num_node_queries,
                query_just=args.query_just,
                query_conf=args.query_conf)

        top_count += 1

        print('Writing queries for top #{} hypothesis with prob {}'.format(
            top_count, prob))

        sparql_helper.execute_sparql_queries(
            node_query_list, stmt_query_list, just_query_list, conf_query_list,
            db_path_list, output_dir,
            filename_prefix='hypothesis-{:0>3d}'.format(top_count),
            header_prefixes=AIF_HEADER_PREFIXES, dry_run=args.dry_run)

        if top_count >= args.top:
            break


if __name__ == '__main__':
    main()
