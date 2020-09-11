"""
Author: Katrin Erk, Mar 2019
- Rule-based creation of initial hypotheses from a JSON variant of a statement of information need.

Update: Pengxiang Cheng, May 2020
- Refactoring and cleanup for dockerization.

Update: Pengxiang Cheng, Aug 2020
- Use new JsonGraph API.

Update: Pengxiang Cheng, Sep 2020
- Adapt to the new behaviors of HypothesisSeedManager, and only save raw seeds per facet to JSON.
"""

import json
import logging
from argparse import ArgumentParser
from typing import Dict

from aida_utexas import util
from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis import HypothesisSeedManager


# helper function for logging
def shortest_name(ere_label: str, json_graph: JsonGraph):
    names = json_graph.english_names(json_graph.ere_names(ere_label))
    if len(names) > 0:
        return sorted(names, key=lambda n: len(n))[0]
    return None


def make_cluster_seeds(json_graph: JsonGraph, query_json: Dict, max_num_seeds_per_facet: int = None,
                       discard_failed_core_constraints: bool = False, rank_cutoff: bool = None):
    # create hypothesis seeds
    logging.info('Making hypothesis seeds for SoIN {} ...'.format(query_json['soin_id']))
    # logging.info('Creating hypothesis seeds .')
    seed_manager = HypothesisSeedManager(
        json_graph=json_graph,
        query_json=query_json,
        discard_failed_core_constraints=discard_failed_core_constraints,
        rank_cutoff=rank_cutoff)

    seeds_by_facet = seed_manager.make_seeds()

    raw_seeds_json = {}

    for facet_label, seeds in seeds_by_facet.items():
        seeds_json = [seed.to_json() for seed in seeds]
        if max_num_seeds_per_facet is not None:
            seeds_json = seeds_json[:max_num_seeds_per_facet]
        raw_seeds_json[facet_label] = seeds_json

    # add graph filename and queries
    raw_seeds_json['graph'] = query_json['graph']

    return raw_seeds_json


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_path', help='Path to the input graph JSON file')
    parser.add_argument('query_path',
                        help='Path to the input query file, or a directory with multiple queries')
    parser.add_argument('output_dir', help='Directory to write the raw hypothesis seeds')
    parser.add_argument('-n', '--max_num_seeds_per_facet', type=int, default=None,
                        help='If provided, only save up to <arg> seeds per facet')
    parser.add_argument('-d', '--discard_failed_core_constraints', action='store_true',
                        help='If specified, discard hypotheses with failed core constraints. '
                             'Try not to use this one during evaluation at first, so that we '
                             'do not discard hypotheses we might still need. If we have too many '
                             'hypotheses and the script runs too slowly, then use this.')
    parser.add_argument('-r', '--rank_cutoff', type=int, default=100,
                        help='If specified, discard hypotheses early if there are at least <arg> '
                             'other hypotheses that have the same fillers for a certain number '
                             '(default = 3) of their non-entrypoint query variables. We might '
                             'need this in the evaluation if some facets have many variables '
                             'that lead to combinatorial explosion.')
    parser.add_argument('-f', '--force', action='store_true',
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    json_graph = JsonGraph.from_dict(util.read_json_file(args.graph_path, 'JSON graph'))

    query_file_paths = util.get_file_list(args.query_path, suffix='.json', sort=True)

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=not args.force)

    for query_file_path in query_file_paths:
        query_json = util.read_json_file(query_file_path, 'query')

        raw_seeds_json = make_cluster_seeds(
            json_graph=json_graph,
            query_json=query_json,
            max_num_seeds_per_facet=args.max_num_seeds_per_facet,
            discard_failed_core_constraints=args.discard_failed_core_constraints,
            rank_cutoff=args.rank_cutoff)

        # write hypotheses out in json format.
        output_path = output_dir / (query_file_path.name.split('_')[0] + '_seeds.json')
        logging.info('Writing raw hypothesis seeds of each facet to {} ...'.format(output_path))
        with open(str(output_path), 'w') as fout:
            json.dump(raw_seeds_json, fout, indent=1)


if __name__ == '__main__':
    main()
