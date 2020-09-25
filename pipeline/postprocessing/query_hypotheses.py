import subprocess
from argparse import ArgumentParser
from operator import itemgetter

from aida_utexas import util
from aida_utexas.aif import JsonGraph, AIDA, LDC, LDC_ONT

AIF_HEADER_PREFIXES = \
    f'@prefix ldcOnt: {LDC_ONT} .\n' \
    f'@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n' \
    f'@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n' \
    f'@prefix aida: {AIDA} .\n' \
    f'@prefix ldc: {LDC} .\n'

QUERY_PREFIXES = \
    f'PREFIX ldcOnt: {LDC_ONT}\n' \
    f'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n' \
    f'PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n' \
    f'PREFIX aida: {AIDA}\n' \
    f'PREFIX ldc: {LDC}\n'


def queries_for_aida_result(
        json_graph, hypothesis, member_to_clusters, cluster_to_prototype, prototype_set,
        num_node_queries=5, num_stmts_per_query=3000,
        query_just=False, query_conf=False):
    ere_id_list = []
    prototype_id_list = []

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
        ere_id_list.append(subject_id)

        if predicate_id != 'type':
            # add object to node_str_list for node query if
            # it's not a typing statement
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
            prototype_id_list.append(prototype_id)

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

    entity_query_item_list, relation_query_item_list, event_query_item_list = set(), set(), set()
    for ere_label in ere_id_list + prototype_id_list:
        if json_graph.is_entity(ere_label):
            entity_query_item_list.add('<{}>'.format(ere_label))
        if json_graph.is_relation(ere_label):
            relation_query_item_list.add('<{}>'.format(ere_label))
        if json_graph.is_event(ere_label):
            event_query_item_list.add('<{}>'.format(ere_label))

    all_query_constraints = []
    if len(entity_query_item_list) > 0:
        all_query_constraints.append('{{?x a aida:Entity .\nFILTER (?x IN ({})) \n}}'.format(
            ', '.join(entity_query_item_list)))
    if len(relation_query_item_list) > 0:
        all_query_constraints.append('{{?x a aida:Relation .\nFILTER (?x IN ({})) \n}}'.format(
            ', '.join(relation_query_item_list)))
    if len(event_query_item_list) > 0:
        all_query_constraints.append('{{?x a aida:Event .\nFILTER (?x IN ({})) \n}}'.format(
            ', '.join(event_query_item_list)))
    if len(cluster_query_item_list) > 0:
        all_query_constraints.append('{{?x a aida:SameAsCluster .\nFILTER (?x IN ({})) \n}}'.format(
            ', '.join(cluster_query_item_list)))

    all_query_constraints.extend(stmt_query_item_list)

    return QUERY_PREFIXES + 'DESCRIBE ?x\nWHERE {{\n{}\n}}'.format(
        '\nUNION\n'.join(all_query_constraints))

    # node_query_item_list = list(set(ere_query_item_list)) + list(set(cluster_query_item_list))
    # stmt_query_item_list = list(set(stmt_query_item_list))
    # just_query_item_list = list(set(just_query_item_list))
    #
    # node_query_list = sparql_helper.produce_node_queries(
    #     node_query_item_list, num_node_queries=1)
    #
    # stmt_query_list = sparql_helper.produce_stmt_queries(
    #     stmt_query_item_list, query_prefixes=QUERY_PREFIXES,
    #     num_stmts_per_query=num_stmts_per_query)

    # just_query_list = sparql_helper.produce_just_queries(
    #     just_query_item_list, query_prefixes=QUERY_PREFIXES,
    #     num_stmts_per_query=num_stmts_per_query)
    #
    # conf_query_list = sparql_helper.produce_conf_queries(
    #     conf_query_item_list, query_prefixes=QUERY_PREFIXES,
    #     num_stmts_per_query=num_stmts_per_query)
    #
    # return node_query_list, stmt_query_list, just_query_list, conf_query_list


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_path', help='path to the graph json file')
    parser.add_argument('hypotheses_path', help='path to the hypotheses json file')
    parser.add_argument('db_dir', help='directory with copies of tdb databases')
    parser.add_argument('output_dir', help='path to output directory')
    parser.add_argument('--top', default=50, type=int,

                        help='number of top hypothesis to output')
    parser.add_argument('--dry_run', action='store_true',
                        help='if specified, only write the SPARQL queries to '
                             'files, without actually executing the queries')
    parser.add_argument('--query_just', action='store_true')
    parser.add_argument('--query_conf', action='store_true')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    json_graph = JsonGraph.from_dict(util.read_json_file(args.graph_path, 'JSON graph'))

    mappings = json_graph.build_cluster_member_mappings()
    member_to_clusters = mappings['member_to_clusters']
    cluster_to_prototype = mappings['cluster_to_prototype']
    prototype_set = set(mappings['prototype_to_clusters'].keys())

    hypotheses_json = util.read_json_file(args.hypotheses_path, 'hypotheses')

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=not args.force)

    db_dir = util.get_input_path(args.db_dir)
    db_path_list = [str(path) for path in sorted(db_dir.glob('copy*'))]
    print('Using the following tdb databases to query: {}'.format(db_path_list))

    num_node_queries = len(db_path_list)

    top_count = 0
    for result_idx, prob in sorted(
            enumerate(hypotheses_json['probs']), key=itemgetter(1), reverse=True):
        hypothesis = hypotheses_json['support'][result_idx]
        # node_query_list, stmt_query_list, just_query_list, conf_query_list = \
        sparql_query_str = \
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

        print(f'Writing queries for hypothesis #{top_count} with prob {prob}')

        sparql_query_path = output_dir / 'hypothesis-{:0>3d}-query.rq'.format(top_count)
        with open(str(sparql_query_path), 'w') as fout:
            fout.write(sparql_query_str + '\n')

        if not args.dry_run:
            query_result_path = output_dir / 'hypothesis-{:0>3d}-raw.ttl'.format(top_count)
            query_cmd = 'echo "query {0}"; tdbquery --loc {1} --query {0} > {2}; '.format(
                sparql_query_path, db_path_list[0], query_result_path)

            print('Executing queries ...')
            process = subprocess.Popen(query_cmd, shell=True)
            process.wait()

        # sparql_helper.execute_sparql_queries(
        #     node_query_list, stmt_query_list, just_query_list, conf_query_list,
        #     db_path_list, output_dir,
        #     filename_prefix='hypothesis-{:0>3d}'.format(top_count),
        #     header_prefixes=AIF_HEADER_PREFIXES, dry_run=args.dry_run)

        if top_count >= args.top:
            break


if __name__ == '__main__':
    main()
