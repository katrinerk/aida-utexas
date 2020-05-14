"""
Author: Katrin Erk October 2018
read aida results json object, along with log object from from coref.py, to produce an
aidaresults.json that looks as if aidabaseline.wppl had run without coref transformation

Update: Pengxiang Cheng August 2019
rewrite for M18 evaluation

Update: Pengxiang Cheng May 2020
slight re-formatting for dockerization
"""

import json
from argparse import ArgumentParser

from aida_utexas import util


def main():
    parser = ArgumentParser()
    parser.add_argument('input_hypotheses_path',
                        help='path to the input json file for hypotheses, or a directory with '
                             'a list of hypotheses files')
    parser.add_argument('output_dir',
                        help='directory to write the coref-recovered hypotheses')
    parser.add_argument('original_graph_path',
                        help='path to the original graph json file')
    parser.add_argument('compressed_graph_path',
                        help='path to the compressed graph json file')
    parser.add_argument('input_log_path',
                        help='path to log file from coref compression')

    args = parser.parse_args()

    input_hypotheses_path = util.get_input_path(args.input_hypotheses_path)
    output_dir = util.get_dir(args.output_dir, create=True)
    original_graph_path = util.get_input_path(args.original_graph_path)
    compressed_graph_path = util.get_input_path(args.compressed_graph_path)
    input_log_path = util.get_input_path(args.input_log_path)

    input_hypotheses_files = util.get_file_list(input_hypotheses_path, suffix='.json', sort=True)

    print('Reading coref log from {}'.format(input_log_path))
    with open(str(input_log_path), 'r') as fin:
        input_log_json = json.load(fin)

    print('Reading original json graph from {}'.format(original_graph_path))
    with open(str(original_graph_path), 'r') as fin:
        original_graph_json = json.load(fin)

    print('Reading compressed json graph from {}'.format(compressed_graph_path))
    with open(str(compressed_graph_path), 'r') as fin:
        compressed_graph_json = json.load(fin)

    for input_hypotheses_file in input_hypotheses_files:
        print('Reading hypotheses from {}'.format(input_hypotheses_file))
        with open(str(input_hypotheses_file), 'r') as fin:
            input_hypotheses_json = json.load(fin)

        # probs do not change
        output_hypotheses_json = {'probs': input_hypotheses_json['probs'], 'support': []}

        for compressed_hypothesis in input_hypotheses_json["support"]:
            original_hypothesis = {'statements': [], 'statementWeights': []}

            # The mapping from each original statement (before coref-compression) to its weight
            original_stmt_weight_mapping = {}

            # Set of cluster membership nodes to include in the original hypothesis
            cluster_membership_set = set()

            for compressed_stmt, stmt_weight in zip(compressed_hypothesis['statements'],
                                                    compressed_hypothesis['statementWeights']):
                # Get the statement entry from the compressed graph
                compressed_stmt_entry = compressed_graph_json['theGraph'][compressed_stmt]
                # Get the cluster(s) from the subject of the compressed statement
                stmt_subj_clusters = \
                    input_log_json['prototype_to_clusters'][compressed_stmt_entry['subject']]
                # Whether this is a type statement
                is_type_stmt = (compressed_stmt_entry['predicate'] == 'type')
                # Get the cluster(s) from the object of the compressed statement if it is an edge
                # statement
                if is_type_stmt:
                    stmt_obj_clusters = None
                else:
                    stmt_obj_clusters = \
                        input_log_json['prototype_to_clusters'][compressed_stmt_entry['object']]

                for original_stmt in input_log_json['new_stmt_to_old_stmts'][compressed_stmt]:
                    # Resolve the statements and weights before coref-compression
                    if original_stmt not in original_stmt_weight_mapping:
                        original_stmt_weight_mapping[original_stmt] = stmt_weight
                    elif original_stmt_weight_mapping[original_stmt] < stmt_weight:
                        original_stmt_weight_mapping[original_stmt] = stmt_weight

                    # Get the statement entry from the original graph
                    original_stmt_entry = original_graph_json['theGraph'][original_stmt]

                    # Add cluster membership between the original subject and each subject cluster
                    stmt_subj = original_stmt_entry['subject']
                    for stmt_subj_cluster in stmt_subj_clusters:
                        cluster_membership_set.add((stmt_subj, stmt_subj_cluster))

                    if is_type_stmt:
                        assert original_stmt_entry['predicate'] == 'type'
                    else:
                        assert original_stmt_entry['predicate'] != 'type'

                        # Add cluster membership between the original object and each object cluster
                        stmt_obj = original_stmt_entry['object']
                        for stmt_obj_cluster in stmt_obj_clusters:
                            cluster_membership_set.add((stmt_obj, stmt_obj_cluster))

            for original_stmt, stmt_weight in original_stmt_weight_mapping.items():
                original_hypothesis['statements'].append(original_stmt)
                original_hypothesis['statementWeights'].append(stmt_weight)

            original_hypothesis['clusterMemberships'] = list(cluster_membership_set)

            original_hypothesis['failedQueries'] = compressed_hypothesis['failedQueries']

            original_query_stmts = set()
            for compressed_query_stmt in compressed_hypothesis['queryStatements']:
                original_query_stmts.update(
                    input_log_json['new_stmt_to_old_stmts'][compressed_query_stmt])
            original_hypothesis['queryStatements'] = list(original_query_stmts)

            output_hypotheses_json['support'].append(original_hypothesis)

        if 'graph' in input_hypotheses_json:
            output_hypotheses_json['graph'] = input_hypotheses_json['graph']
        if 'queries' in input_hypotheses_json:
            output_hypotheses_json['queries'] = input_hypotheses_json['queries']

        output_hypotheses_path = util.get_output_path(output_dir / input_hypotheses_file.name)
        print('Writing coref-recovered hypotheses to {}'.format(output_hypotheses_path))
        with open(str(output_hypotheses_path), 'w') as fout:
            json.dump(output_hypotheses_json, fout, indent=2)


if __name__ == '__main__':
    main()
