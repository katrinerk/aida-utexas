"""
Author: Pengxiang Cheng August 2019

Construct sub-graphs for each hypothesis seed by starting at the entry points and go up to k hops
"""

import argparse
import json
import logging
from copy import deepcopy

from aida_utexas import util


def stmts_to_eres(graph, hop_idx, this_hop_stmts, nodes_so_far, verbose=False):
    this_hop_general_stmts = set()
    this_hop_typing_stmts = set()

    this_hop_eres = set()
    this_hop_entities = set()
    this_hop_relations = set()
    this_hop_events = set()

    def _process_stmt_arg(stmt_arg):
        this_hop_eres.add(stmt_arg)
        arg_node = graph['theGraph'][stmt_arg]
        if arg_node['type'] == 'Entity':
            this_hop_entities.add(stmt_arg)
        elif arg_node['type'] == 'Relation':
            this_hop_relations.add(stmt_arg)
        else:
            assert arg_node['type'] == 'Event'
            this_hop_events.add(stmt_arg)

    for stmt_label in this_hop_stmts:
        stmt_node = graph['theGraph'][stmt_label]

        stmt_subj = stmt_node.get('subject', None)
        stmt_obj = stmt_node.get('object', None)

        assert stmt_subj is not None and stmt_subj in graph['theGraph']
        if stmt_subj not in nodes_so_far['eres']:
            _process_stmt_arg(stmt_subj)

        assert stmt_obj is not None
        if stmt_obj in graph['theGraph']:
            this_hop_general_stmts.add(stmt_label)

            if stmt_obj not in nodes_so_far['eres']:
                _process_stmt_arg(stmt_obj)

        elif stmt_node.get('predicate', None) == 'type':
            # assert stmt_node.get('predicate', None) == 'type', stmt_node
            this_hop_typing_stmts.add(stmt_label)

    nodes_so_far['stmts'].update(this_hop_stmts)
    nodes_so_far['general_stmts'].update(this_hop_general_stmts)
    nodes_so_far['typing_stmts'].update(this_hop_typing_stmts)

    nodes_so_far['eres'].update(this_hop_eres)
    nodes_so_far['entities'].update(this_hop_entities)
    nodes_so_far['relations'].update(this_hop_relations)
    nodes_so_far['events'].update(this_hop_events)

    if verbose:
        print('\n\tIn hop-{}'.format(hop_idx))
        print('\tFound {} general statements (cumulative = {})'.format(
            len(this_hop_general_stmts), len(nodes_so_far['general_stmts'])))
        print('\tFound {} typing statements (cumulative = {})'.format(
            len(this_hop_typing_stmts), len(nodes_so_far['typing_stmts'])))
        print('\tFound {} entities (cumulative = {})'.format(
            len(this_hop_entities), len(nodes_so_far['entities'])))
        print('\tFound {} relations (cumulative = {})'.format(
            len(this_hop_relations), len(nodes_so_far['relations'])))
        print('\tFound {} events (cumulative = {})'.format(
            len(this_hop_events), len(nodes_so_far['events'])))

    return this_hop_eres


def nodes_to_subgraph(graph, nodes_dict):
    subgraph = {
        'theGraph': {},
        'ere': [],
        'statements': []
    }

    for ere_label in nodes_dict['eres']:
        ere_entry = deepcopy(graph['theGraph'][ere_label])

        assert ere_entry['type'] in ['Entity', 'Relation', 'Event']

        adjacent_stmts = [
            stmt_label for stmt_label in ere_entry['adjacent']
            if stmt_label in nodes_dict['stmts']
        ]
        ere_entry['adjacent'] = adjacent_stmts

        subgraph['theGraph'][ere_label] = ere_entry
        subgraph['ere'].append(ere_label)

    for stmt_label in nodes_dict['stmts']:
        stmt_entry = deepcopy(graph['theGraph'][stmt_label])

        assert stmt_entry['type'] == 'Statement'
        if stmt_entry['subject'] not in subgraph['theGraph']:
            continue
        if stmt_entry['predicate'] != 'type':
            if stmt_entry['object'] not in subgraph['theGraph']:
                continue

        subgraph['theGraph'][stmt_label] = stmt_entry
        subgraph['statements'].append(stmt_label)

    return subgraph


def extract_subgraph(index, graph, statements, max_num_hops, min_num_eres, min_num_stmts,
                     verbose=False):
    nodes_so_far = {
        'stmts': set(),
        'general_stmts': set(),
        'typing_stmts': set(),
        'eres': set(),
        'entities': set(),
        'relations': set(),
        'events': set()
    }

    zero_hop_stmts = set(statements)

    last_hop_eres = stmts_to_eres(
        graph=graph, hop_idx=0, this_hop_stmts=zero_hop_stmts,
        nodes_so_far=nodes_so_far, verbose=verbose)

    num_hops = 1
    while True:
        if max_num_hops is not None and num_hops >= max_num_hops:
            break

        num_hops += 1

        this_hop_stmts = set()

        for ere_label in last_hop_eres:
            ere_node = graph['theGraph'][ere_label]
            for stmt_label in ere_node['adjacent']:
                if stmt_label not in nodes_so_far['stmts'] and \
                        stmt_label in graph['theGraph']:
                    this_hop_stmts.add(stmt_label)

        last_hop_eres = stmts_to_eres(
            graph=graph, hop_idx=num_hops, this_hop_stmts=this_hop_stmts,
            nodes_so_far=nodes_so_far, verbose=verbose)

        if min_num_eres is not None or min_num_stmts is not None:
            stop = True
            if min_num_eres is not None and len(nodes_so_far['eres']) < min_num_eres:
                stop = False
            if min_num_stmts is not None and len(nodes_so_far['stmts']) < min_num_stmts:
                stop = False
            if stop:
                break

    extra_typing_stmts = set()
    for ere_label in last_hop_eres:
        ere_node = graph['theGraph'][ere_label]
        for stmt_label in ere_node['adjacent']:
            if stmt_label not in nodes_so_far['stmts'] and \
                    stmt_label in graph['theGraph']:
                stmt_node = graph['theGraph'][stmt_label]

                stmt_subj = stmt_node.get('subject', None)
                stmt_pred = stmt_node.get('predicate', None)
                if stmt_subj in last_hop_eres and stmt_pred == 'type':
                    extra_typing_stmts.add(stmt_label)

    nodes_so_far['stmts'].update(extra_typing_stmts)
    nodes_so_far['typing_stmts'].update(extra_typing_stmts)

    if verbose:
        print('\n\tAfter hop-{}'.format(num_hops))
        print('\tFound {} extra typing statements (cumulative = {})'.format(
            len(extra_typing_stmts), len(nodes_so_far['typing_stmts'])))

    logging.info(
        'Extracting subgraph for hypothesis # {}: found {} EREs and {} statements '
        'after {} hops'.format(
            index + 1, len(nodes_so_far['eres']), len(nodes_so_far['stmts']), num_hops))

    subgraph = nodes_to_subgraph(graph, nodes_so_far)
    return subgraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_file', help='path to the graph json file')
    parser.add_argument('seed_file', help='path to the hypothesis seed file')
    parser.add_argument('output_dir', help='path to the output directory')
    parser.add_argument('--max_num_hops', type=int, default=None,
                        help='maximum number of hops to extend from')
    parser.add_argument('--min_num_eres', type=int, default=None,
                        help='minimum number of EREs in the subgraph to stop extending')
    parser.add_argument('--min_num_stmts', type=int, default=None,
                        help='minimum number of statements in the subgraph to stop extending')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='print more details in each hop of extraction')
    parser.add_argument('--force', '-f', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=not args.force)

    graph_json = util.read_json_file(args.graph_file, 'JSON graph')
    seed_json = util.read_json_file(args.seed_file, 'hypothesis seeds')

    max_num_hops = args.max_num_hops
    min_num_eres = args.min_num_eres
    min_num_stmts = args.min_num_stmts

    if not (max_num_hops or min_num_eres or min_num_stmts):
        raise RuntimeError('Must specify at least one of "max_num_hops", "min_num_eres", and '
                           '"min_num_stmts"')

    for hypothesis_idx, (prob, hypothesis) in enumerate(
            zip(seed_json['probs'], seed_json['support'])):
        subgraph = extract_subgraph(
            index=hypothesis_idx,
            graph=graph_json,
            statements=hypothesis['statements'],
            max_num_hops=max_num_hops,
            min_num_eres=min_num_eres,
            min_num_stmts=min_num_stmts,
            verbose=args.verbose
        )
        output_path = output_dir / f'subgraph_{hypothesis_idx}.json'
        with open(str(output_path), 'w') as fout:
            json.dump(subgraph, fout, indent=2)


if __name__ == '__main__':
    main()
