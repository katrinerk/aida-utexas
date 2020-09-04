"""
Author: Katrin Erk October 2018
- Resolve all coreferences in an aidagraph.json data structure, and write a reduced aidagraph.json
data structure without coreferring EREs or statements, along with a second .json file that details
which original EREs/statements correspond to which reduced ones.

Update: Pengxiang Cheng, August 2019
- Rewrite for M18 evaluation.

Update: Pengxiang Cheng, May 2020
- Slight re-formatting for dockerization.

Update: Pengxiang Cheng, August 2020
- Adapt to new JsonGraph APIs.
"""

import itertools
import json
import logging
from argparse import ArgumentParser
from collections import defaultdict, Counter
from copy import deepcopy
from typing import Dict

from aida_utexas import util
from aida_utexas.aif import JsonGraph
from aida_utexas.aif.json_graph import ERENode, StatementNode
from aida_utexas.json_graph_helper import build_cluster_member_mappings


def make_stmt_keys(stmt_entry, member_to_prototypes):
    subj = stmt_entry.subject
    if subj in member_to_prototypes:
        new_subj_set = member_to_prototypes[subj]
    else:
        logging.warning('Warning: statement subject {} not found in '
                        'any ClusterMembership node'.format(subj))
        new_subj_set = {subj}

    pred = stmt_entry.predicate

    obj = stmt_entry.object
    if pred != 'type':
        if obj in member_to_prototypes:
            new_obj_set = member_to_prototypes[obj]
        else:
            logging.warning('Warning: statement object {} not found in '
                            'any ClusterMembership node'.format(obj))
            new_obj_set = {obj}
    else:
        new_obj_set = {obj}

    return [(new_subj, pred, new_obj) for new_subj, new_obj
            in itertools.product(new_subj_set, new_obj_set)]


def build_mappings(json_graph: JsonGraph):
    # Build mappings among clusters, members, and prototypes
    mappings = build_cluster_member_mappings(json_graph)

    # Build mappings from old statement labels to new statement labels
    stmt_count = 0

    stmt_key_to_new_stmt = {}
    new_stmt_to_stmt_key = {}
    old_stmt_to_new_stmts = defaultdict(set)
    new_stmt_to_old_stmts = defaultdict(set)

    for node_label, node in json_graph.node_dict.items():
        if node.type == 'Statement':
            stmt_keys = make_stmt_keys(
                stmt_entry=node, member_to_prototypes=mappings['member_to_prototypes'])
            for stmt_key in stmt_keys:
                if stmt_key not in stmt_key_to_new_stmt:
                    new_stmt_label = 'Statement-{}'.format(stmt_count)
                    stmt_count += 1
                    stmt_key_to_new_stmt[stmt_key] = new_stmt_label
                    new_stmt_to_stmt_key[new_stmt_label] = stmt_key
                else:
                    new_stmt_label = stmt_key_to_new_stmt[stmt_key]

                old_stmt_to_new_stmts[node_label].add(new_stmt_label)
                new_stmt_to_old_stmts[new_stmt_label].add(node_label)

    num_old_stmts = len(old_stmt_to_new_stmts)
    num_new_stmts = len(new_stmt_to_old_stmts)

    assert len(stmt_key_to_new_stmt) == num_new_stmts
    assert len(new_stmt_to_stmt_key) == num_new_stmts

    print('\nConstructed mapping from {} old statements to {} new statements'.format(
        num_old_stmts, num_new_stmts))

    new_stmts_per_old_stmt_counter = Counter(
        [len(v) for v in old_stmt_to_new_stmts.values()])
    for key in sorted(new_stmts_per_old_stmt_counter.keys()):
        if key > 1:
            print('\tFor {} out of {} old statements, each is mapped to {} new statements'.format(
                new_stmts_per_old_stmt_counter[key], num_old_stmts, key))

    mappings.update({
        'stmt_key_to_new_stmt': stmt_key_to_new_stmt,
        'new_stmt_to_stmt_key': new_stmt_to_stmt_key,
        'old_stmt_to_new_stmts': old_stmt_to_new_stmts,
        'new_stmt_to_old_stmts': new_stmt_to_old_stmts
    })

    return mappings


def compress_eres(input_json_graph: JsonGraph, mappings: Dict, output_json_graph: JsonGraph):
    assert len(output_json_graph.eres) == 0

    logging.info('Building ERE / SameAsCluster / ClusterMembership entries for the compressed '
                 'graph ...')

    num_new_eres = 0

    for prototype, members in mappings['prototype_to_members'].items():
        old_entry = input_json_graph.node_dict[prototype]

        # Use the same ERE index from the original graph
        new_entry = {'index': old_entry.index}

        member_entry_list = [input_json_graph.node_dict[member] for member in members]

        # Resolve the type of the compressed ERE node
        type_set = set(member_entry.type for member_entry in member_entry_list)
        # if len(type_set) > 1:
        #     type_set.remove('Entity')
        if len(type_set) > 1:
            logging.warning('Error: multiple types {} from the following EREs {}'.format(
                type_set, members))
        new_entry['type'] = type_set.pop()

        # Resolve the adjacent statements of the compressed ERE node
        adjacency_set = set()
        for member_entry in member_entry_list:
            for old_stmt in member_entry.adjacent:
                adjacency_set.update(
                    mappings['old_stmt_to_new_stmts'][old_stmt])
        new_entry['adjacent'] = list(adjacency_set)

        # Resolve the names of the compressed ERE node
        name_set = set()
        for member_entry in member_entry_list:
            name_set.update(member_entry.name)
        for cluster in mappings['prototype_to_clusters'][prototype]:
            cluster_handle = input_json_graph.node_dict[cluster].handle
            if cluster_handle is not None and cluster_handle != '[unknown]':
                name_set.add(cluster_handle)
        new_entry['name'] = list(name_set)

        # Resolve the LDC time list of the compressed ERE node
        ldc_time_list = []
        for member_entry in member_entry_list:
            ldc_time_list.extend(member_entry.ldcTime)
        new_entry['ldcTime'] = ldc_time_list

        output_json_graph.node_dict[prototype] = ERENode(**new_entry)
        output_json_graph.eres.append(prototype)

        # Add SameAsCluster nodes and ClusterMembership nodes
        for cluster in mappings['prototype_to_clusters'][prototype]:
            output_json_graph.node_dict[cluster] = deepcopy(input_json_graph.node_dict[cluster])

            for cluster_membership_key in \
                    mappings['cluster_membership_key_mapping'][(cluster, prototype)]:
                output_json_graph.node_dict[cluster_membership_key] = deepcopy(
                    input_json_graph.node_dict[cluster_membership_key])

        num_new_eres += 1

    return num_new_eres


def compress_statements(input_json_graph: JsonGraph, mappings: Dict, output_json_graph: JsonGraph):
    logging.info('Building statement entries for the compressed graph ...')

    assert len(output_json_graph.statements) == 0

    num_new_stmts = 0

    for new_stmt, stmt_key in mappings['new_stmt_to_stmt_key'].items():
        stmt_idx = int(new_stmt.split('-')[1])
        subj, pred, obj = stmt_key
        new_entry = {
            'type': 'Statement',
            'index': stmt_idx,
            'subject': subj,
            'predicate': pred,
            'object': obj
        }

        conf_levels = set()
        for old_stmt in mappings['new_stmt_to_old_stmts'][new_stmt]:
            old_stmt_entry = input_json_graph.node_dict[old_stmt]
            if old_stmt_entry.conf is not None:
                conf_levels.add(old_stmt_entry.conf)

        new_entry['conf'] = max(conf_levels) if conf_levels else None

        # old_stmt_entry_list = [input_json_graph.node_dict[old_stmt]
        #                        for old_stmt in mappings['new_stmt_to_old_stmts'][new_stmt]]
        #
        # # Resolve the extra information (source and hypotheses) of the new
        # # statement
        # for label in ['source', 'hypotheses_supported', 'hypotheses_partially_supported',
        #               'hypotheses_contradicted']:
        #     label_value_set = set()
        #     for old_stmt_entry in old_stmt_entry_list:
        #         if label in old_stmt_entry:
        #             label_value_set.update(old_stmt_entry[label])
        #     if len(label_value_set) > 0:
        #         new_entry[label] = list(label_value_set)

        output_json_graph.node_dict[new_stmt] = StatementNode(**new_entry)
        output_json_graph.statements.append(new_stmt)
        num_new_stmts += 1

    return num_new_stmts


def main():
    parser = ArgumentParser()
    parser.add_argument('input_graph_path', help='path to the input graph json file')
    parser.add_argument('output_graph_path', help='path to write the coref-compressed graph')
    parser.add_argument('output_log_path', help='path to write the log file')

    args = parser.parse_args()

    input_graph_path = util.get_input_path(args.input_graph_path)
    output_graph_path = util.get_output_path(args.output_graph_path)
    output_log_path = util.get_output_path(args.output_log_path)

    logging.info('Reading json graph from {} ...'.format(input_graph_path))
    with open(str(input_graph_path), 'r') as fin:
        input_json_graph = JsonGraph.from_dict(json.load(fin))

    num_old_eres = len(list(input_json_graph.each_ere()))
    assert num_old_eres == len(input_json_graph.eres)
    num_old_stmts = len(list(input_json_graph.each_statement()))
    logging.info('Found {} EREs and {} statements in the original graph'.format(
        num_old_eres, num_old_stmts))

    mappings = build_mappings(input_json_graph)

    output_json_graph = JsonGraph()

    num_new_eres = compress_eres(input_json_graph, mappings, output_json_graph)
    num_new_stmts = compress_statements(input_json_graph, mappings, output_json_graph)

    logging.info('Finished coref-compressed graph with {} EREs and {} statements'.format(
        num_new_eres, num_new_stmts))

    logging.info('Writing compressed json graph to {}'.format(output_graph_path))
    with open(str(output_graph_path), 'w') as fout:
        json.dump(output_json_graph.as_dict(), fout, indent=1)

    log_json = {}
    for mapping_key, mapping in mappings.items():
        if 'key' in mapping_key:
            continue
        if mapping_key.endswith('s'):
            log_json[mapping_key] = {k: list(v) for k, v in mapping.items()}
        else:
            log_json[mapping_key] = mapping

    logging.info('Writing compression log to {}'.format(output_log_path))
    with open(str(output_log_path), 'w') as fout:
        json.dump(log_json, fout, indent=2)


if __name__ == '__main__':
    main()
