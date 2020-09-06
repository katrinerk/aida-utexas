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
    parser.add_argument('hypotheses_path',
                        help='Path to the raw hypotheses file, or a directory with multiple files')
    parser.add_argument('output_dir',
                        help='Directory to write the filtered hypothesis files(s)')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=not args.force)

    json_graph = JsonGraph.from_dict(util.read_json_file(args.graph_path, 'JSON graph'))
    hypotheses_file_paths = util.get_file_list(args.hypotheses_path, suffix='.json', sort=True)

    for hypotheses_file_path in hypotheses_file_paths:
        json_hypotheses = util.read_json_file(hypotheses_file_path, 'hypotheses')
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

        output_path = output_dir / hypotheses_file_path.name
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
