"""
Author: Katrin Erk March 2019
rule-based creation of initial hypotheses from a JSON variant of a statement of information need

Update: Pengxiang Cheng May 2020
refactoring and cleanup for dockerization
"""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

from aida_utexas import util
from aida_utexas.aif import AidaJson
from aida_utexas.seeds import ClusterExpansion, ClusterSeeds


# helper function for logging
def shortest_name(label: str, graph_json: AidaJson):
    if label in graph_json.thegraph and 'name' in graph_json.thegraph[label]:
        names = graph_json.english_names(graph_json.thegraph[label]['name'])
        if len(names) > 0:
            return sorted(names, key=lambda s: len(s))[0]
    return None


def make_cluster_seeds(graph_json: AidaJson, query_json: Dict, output_path: Path, graph_name: str,
                       max_num_seeds: int = None, discard_failed_queries: bool = False,
                       early_cutoff: bool = False, rank_cutoff: bool = None, log: bool = False):
    # create cluster seeds
    logging.info('Creating cluster seeds .')
    cluster_seeds = ClusterSeeds(graph_json, query_json,
                                 discard_failedqueries=discard_failed_queries,
                                 earlycutoff=early_cutoff,
                                 qs_cutoff=rank_cutoff)

    # and expand on them
    logging.info('Expanding cluster seeds ...')
    cluster_expansion = ClusterExpansion(graph_json, cluster_seeds.finalize())
    cluster_expansion.type_completion()
    cluster_expansion.affiliation_completion()

    json_seeds = cluster_expansion.to_json()

    # possibly prune seeds
    if max_num_seeds is not None:
        json_seeds['probs'] = json_seeds['probs'][:max_num_seeds]
        json_seeds['support'] = json_seeds['support'][:max_num_seeds]

    # add graph filename and queries
    json_seeds['graph'] = graph_name
    json_seeds['queries'] = query_json['queries']

    # write hypotheses out in json format.
    logging.info('Writing seeds to {} ...'.format(output_path))
    with open(str(output_path), 'w') as fout:
        json.dump(json_seeds, fout, indent=1)

    if log:
        log_path = output_path.with_suffix('.log')

        with open(str(log_path), 'w') as fout:
            print('Number of hypotheses:', len(json_seeds['support']), file=fout)
            print('Number of hypotheses without failed queries:',
                  len([h for h in json_seeds['support'] if len(h['failedQueries']) == 0]),
                  file=fout)

            for hyp in cluster_expansion.hypotheses()[:10]:
                print('hypothesis log wt', hyp.lweight, file=fout)
                for qvar, filler in sorted(hyp.qvar_filler.items()):
                    name = shortest_name(filler, graph_json)
                    if name is not None:
                        print(qvar, ':', filler, name, file=fout)
                    else:
                        print(qvar, ':', filler, file=fout)

                print('\n', file=fout)


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_path', help='Path to the input graph JSON file')
    parser.add_argument('query_path',
                        help='Path to the input query file, or a directory with multiple queries')
    parser.add_argument('output_dir',
                        help='Directory to write the cluster seeds')
    # maximum number of seeds to store. Do use this during evaluation if we get lots of cluster
    # seeds! We will only get evaluated on a limited number of top hypotheses anyway.
    parser.add_argument('-n', '--max_num_seeds', type=int, default=None,
                        help='only list up to n cluster seeds')
    # discard hypotheses with failed queries? Try not to use this one during evaluation at first,
    # so that we don't discard hypotheses we might still need. If we have too many hypotheses and
    # the script runs too slowly, then use this.
    parser.add_argument('-f', '--discard_failed_queries', action='store_true', default=False,
                        help='discard hypotheses that have failed queries')
    # early cutoff: discard queries below the best k based only on entry point scores. Try not to
    # use this one during evaluation at first, so that we don't discard hypotheses we might still
    # need. If we have too many hypotheses and the script runs too slowly, then use this.
    parser.add_argument('-c', '--early_cutoff', type=int, default=None,
                        help='discard hypotheses below the best n based only on entry point scores')

    # rank-based cutoff: discards hypotheses early if there are at least <arg> other hypotheses
    # that coincide with this one in 3 query variable fillers. We do need this in the evaluation!
    # Otherwise combinatorial explosion happens. I've standard-set this to 100.
    parser.add_argument('-r', '--rank_cutoff', type=int, default=100,
                        help='discard hypotheses early if there are n others that have the same '
                             'fillers for 3 of their query variables')
    # write logs? Do this for qualitative analysis of the diversity of query responses, but do not
    # use this during evaluation, as it slows down the script.
    parser.add_argument('-l', '--log', action='store_true', default=False,
                        help='write log files to output directory')

    args = parser.parse_args()

    graph_path = util.get_input_path(args.graph_path)
    query_path = util.get_input_path(args.query_path)
    output_dir = util.get_dir(args.output_dir, create=True)

    graph_name = graph_path.name
    logging.info('Loading graph JSON from {} ...'.format(graph_path))
    with open(str(graph_path), 'r') as fin:
        graph_json = AidaJson(json.load(fin))

    query_file_paths = util.get_file_list(query_path, suffix='.json', sort=True)

    for query_file_path in query_file_paths:
        logging.info('Processing query: {} ...'.format(query_file_path))
        with open(str(query_file_path), 'r') as fin:
            query_json = json.load(fin)

        query_name = query_file_path.name
        output_path = util.get_output_path(output_dir / (query_name.split('_')[0] + '_seeds.json'))

        make_cluster_seeds(graph_json=graph_json,
                           query_json=query_json,
                           output_path=output_path,
                           graph_name=graph_name,
                           max_num_seeds=args.max_num_seeds,
                           discard_failed_queries=args.discard_failed_queries,
                           early_cutoff=args.early_cutoff,
                           rank_cutoff=args.rank_cutoff,
                           log=args.log)


if __name__ == '__main__':
    main()
