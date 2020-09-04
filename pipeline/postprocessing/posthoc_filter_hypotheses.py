"""
Author: Katrin Erk, Jul 2019
- Simple post-hoc filter for hypothesis files

Update: Pengxiang Cheng, May 2020
- Re-writing for dockerization

Update: Pengxiang Cheng, Aug 2020
- Use the new JsonGraph API
"""

import json
import logging
from argparse import ArgumentParser

from aida_utexas import util
from aida_utexas.aif import JsonGraph
from aida_utexas.seeds import AidaHypothesisCollection, AidaHypothesisFilter, ClusterExpansion


def main():
    parser = ArgumentParser()

    parser.add_argument('graph_path', help='Path to the input graph JSON file')
    parser.add_argument('hypothesis_path',
                        help='Path to the raw hypothesis file, or a directory with multiple files')
    parser.add_argument('output_dir',
                        help='Directory to write the filtered hypothesis files(s)')

    args = parser.parse_args()

    graph_path = util.get_input_path(args.graph_path)
    hypothesis_path = util.get_input_path(args.hypothesis_path)
    output_dir = util.get_dir(args.output_dir, create=True)

    logging.info('Loading graph JSON from {} ...'.format(graph_path))
    with open(str(graph_path), 'r') as fin:
        json_graph = JsonGraph.from_dict(json.load(fin))

    hypothesis_file_paths = util.get_file_list(hypothesis_path, suffix='.json', sort=True)

    for hypothesis_file_path in hypothesis_file_paths:
        logging.info('Processing hypotheses from {} ...'.format(hypothesis_file_path))

        with open(str(hypothesis_file_path), 'r') as fin:
            json_hypotheses = json.load(fin)
        hypothesis_collection = AidaHypothesisCollection.from_json(json_hypotheses, json_graph)

        # create the filter
        hypothesis_filter = AidaHypothesisFilter(json_graph)

        cluster_expansion = ClusterExpansion(json_graph, hypothesis_collection)
        cluster_expansion.type_completion()
        cluster_expansion.affiliation_completion()

        new_hypothesis_collection = AidaHypothesisCollection([])

        for hypothesis in hypothesis_collection.hypotheses:
            new_hypothesis = hypothesis_filter.filtered(hypothesis)
            new_hypothesis_collection.add(new_hypothesis)

        output_path = util.get_output_path(output_dir / hypothesis_file_path.name)
        logging.info('Writing filtered hypotheses to {} ...'.format(output_path))

        with open(str(output_path), 'w') as fout:
            new_json_hypotheses = new_hypothesis_collection.to_json()

            # add graph filename and queries, if they were there before
            if 'graph' in json_hypotheses:
                new_json_hypotheses['graph'] = json_hypotheses['graph']
            if "queries" in json_hypotheses:
                new_json_hypotheses['queries'] = json_hypotheses['queries']

            json.dump(new_json_hypotheses, fout, indent=1)


if __name__ == '__main__':
    main()
