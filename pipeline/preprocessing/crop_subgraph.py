"""
Author: Pengxiang Cheng August 2019

Construct sub-graphs for each cluster seed by starting at the entry points and
"""

import argparse
import json
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

        else:
            assert stmt_node.get('predicate', None) == 'type', stmt_node
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
        assert stmt_entry['subject'] in subgraph['theGraph']
        if stmt_entry['predicate'] != 'type':
            assert stmt_entry['object'] in subgraph['theGraph']

        subgraph['theGraph'][stmt_label] = stmt_entry
        subgraph['statements'].append(stmt_label)

    return subgraph


def extract_subgraph(graph, statements, num_hops=2, verbose=False):
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

    for hop_idx in range(1, num_hops + 1):
        this_hop_stmts = set()

        for ere_label in last_hop_eres:
            ere_node = graph['theGraph'][ere_label]
            for stmt_label in ere_node['adjacent']:
                if stmt_label not in nodes_so_far['stmts'] and \
                        stmt_label in graph['theGraph']:
                    this_hop_stmts.add(stmt_label)

        last_hop_eres = stmts_to_eres(
            graph=graph, hop_idx=hop_idx, this_hop_stmts=this_hop_stmts,
            nodes_so_far=nodes_so_far, verbose=verbose)

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

    print('\tFound {} EREs and {} statements after {} hops'.format(
        len(nodes_so_far['eres']), len(nodes_so_far['stmts']), num_hops))

    subgraph = nodes_to_subgraph(graph, nodes_so_far)
    return subgraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_file', help='path to the graph json file')
    parser.add_argument('seed_file', help='path to the cluster seed file')
    parser.add_argument('output_dir', help='path to the output directory')
    parser.add_argument('--num_hops', '-n', type=int, default=2,
                        help='number of hops to extend from')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='print more details in each hop of extraction')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=not args.force)

    graph_json = util.read_json_file(args.graph_file, 'JSON graph')
    seed_json = util.read_json_file(args.seed_file, 'cluster seeds')

    for hypothesis_idx, (prob, hypothesis) in enumerate(
            zip(seed_json['probs'], seed_json['support'])):
        print('\nExtracting subgraph for hypothesis # {} with prob = {}'.format(
            hypothesis_idx, prob))
        subgraph = extract_subgraph(
            graph=graph_json,
            statements=hypothesis['statements'],
            num_hops=args.num_hops,
            verbose=args.verbose
        )
        output_path = output_dir / f'subgraph_{hypothesis_idx}.json'
        print('Writing subgraph json to {}'.format(output_path))
        with open(str(output_path), 'w') as fout:
            json.dump(subgraph, fout, indent=2)


if __name__ == '__main__':
    main()
