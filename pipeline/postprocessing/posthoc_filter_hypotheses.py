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
from aida_utexas.hypothesis import AidaHypothesisCollection, AidaHypothesisFilter


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
        hypotheses_json = util.read_json_file(hypotheses_file_path, 'hypotheses')
        hypothesis_collection = AidaHypothesisCollection.from_json(hypotheses_json, json_graph)

        hypothesis_collection.expand()

        # create the filter
        hypothesis_filter = AidaHypothesisFilter(json_graph)

        filtered_hypothesis_collection = AidaHypothesisCollection(
            [hypothesis_filter.filtered(hypothesis) for hypothesis in hypothesis_collection])

        filtered_hypotheses_json = filtered_hypothesis_collection.to_json()

        # add graph filename and queries, if they were there before
        if 'graph' in hypotheses_json:
            filtered_hypotheses_json['graph'] = hypotheses_json['graph']
        if "queries" in hypotheses_json:
            filtered_hypotheses_json['queries'] = hypotheses_json['queries']

        output_path = output_dir / hypotheses_file_path.name
        logging.info('Writing filtered hypotheses to {} ...'.format(output_path))

        with open(str(output_path), 'w') as fout:
            json.dump(filtered_hypotheses_json, fout, indent=1)


if __name__ == '__main__':
    main()
