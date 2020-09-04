"""
Author: Katrin Erk, Oct 2018
- Generates a json file called aidagraph.json that describes an AIDA graph, the units for
clustering, and baseline values for cluster distances.

Update: Pengxiang Cheng, Aug 2019
- Rewrite for M18 evaluation

Update: Pengxiang Cheng, May 2020
- Incorporate the SoIN processing script into this file to avoid loading the same TTL twice.

Update: Pengxiang Cheng, Aug 2020
- Change to use new APIs from aida_utexas.aif and aida_utexas.soin
"""

import json
import logging
from argparse import ArgumentParser

from aida_utexas import util
from aida_utexas.aif import AidaGraph, JsonGraph
from aida_utexas.soin import SOIN, get_cluster_mappings, resolve_all_entrypoints

duplicate_kb_file = 'resources/duplicate_kb_id_mapping.json'


def main():
    parser = ArgumentParser(
        description='Read in a TA2 KB and a (list of) XML-based Statement of Information Need '
                    'definition, convert the KB to JSON format, then convert each SoIN to a JSON '
                    'query by identifying and ranking entry points.')
    parser.add_argument('kb_path', help='Path to the input TA2 KB')
    parser.add_argument('soin_path',
                        help='Path to the input SoIN file, or a directory with multiple SoIN files')
    parser.add_argument('graph_output_path', help='Path to write the JSON graph')
    parser.add_argument('query_output_dir', help='Directory to write the JSON queries')
    parser.add_argument('-m', '--max_matches', type=int, default=50,
                        help='The maximum number of EPs *per entry point description*')
    parser.add_argument('-d', '--dup_kb', default=duplicate_kb_file,
                        help='Path to the json file with duplicate KB ID mappings')
    parser.add_argument('-f', '--force_overwrite', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    kb_path = util.get_input_path(args.kb_path)
    soin_path = util.get_input_path(args.soin_path)
    graph_output_path = util.get_output_path(
        args.graph_output_path, overwrite_warning=not args.force_overwrite)
    query_output_dir = util.get_dir(args.query_output_dir, create=True)

    aida_graph = AidaGraph()
    aida_graph.build_graph(str(kb_path), fmt='ttl')

    json_graph = JsonGraph()
    json_graph.build_graph(aida_graph)

    logging.info('Writing JSON graph to {} ...'.format(graph_output_path))
    with open(str(graph_output_path), 'w') as fout:
        json.dump(json_graph.as_dict(), fout, indent=1)
    logging.info('Done.')

    soin_file_paths = util.get_file_list(soin_path, suffix='.xml', sort=True)

    dup_kb_id_mapping = None
    if args.dup_kb is not None:
        logging.info('Loading duplicate KB ID mapping from {} ...'.format(args.dup_kb))
        with open(args.dup_kb, 'r') as fin:
            dup_kb_id_mapping = json.load(fin)

    logging.info('Getting Cluster Mappings ...')
    cluster_to_prototype, entity_to_clusters = get_cluster_mappings(aida_graph)

    for soin_file_path in soin_file_paths:
        query_output_path = util.get_output_path(
            query_output_dir / (soin_file_path.stem + '_query.json'),
            overwrite_warning=not args.force_overwrite)

        logging.info('Processing SOIN {} ...'.format(soin_file_path))
        logging.info('Parsing SOIN XML ...')
        soin = SOIN.parse(str(soin_file_path), dup_kbid_mapping=dup_kb_id_mapping)

        logging.info('Resolving all entrypoints ...')
        ep_matches_dict, ep_weights_dict = resolve_all_entrypoints(
            graph=aida_graph,
            soin=soin,
            cluster_to_prototype=cluster_to_prototype,
            entity_to_clusters=entity_to_clusters,
            max_matches=args.max_matches)

        logging.info('Serializing data structures ...')
        query_json = {
            'graph': kb_path.stem,
            'soin_id': soin.id,
            'frame_id': [frame.id for frame in soin.frames],
            'entrypoints': ep_matches_dict,
            'entrypointWeights': ep_weights_dict,
            'queries': [],
            'facets': soin.frames_to_json(),
        }

        logging.info('Writing JSON query to {} ...'.format(query_output_path))
        with open(str(query_output_path), 'w') as fout:
            json.dump(query_json, fout, indent=1)
        logging.info('Done.')


if __name__ == '__main__':
    main()
