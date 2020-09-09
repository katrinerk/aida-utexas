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

Update: Pengxiang Cheng, Sep 2020
- Merge the get_cluster_mappings method from process_soin
"""

import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict

from aida_utexas import util
from aida_utexas.aif import AidaGraph, JsonGraph
from aida_utexas.soin import SOIN

duplicate_kb_file = 'resources/duplicate_kb_id_mapping.json'


def get_cluster_mappings(aida_graph: AidaGraph) -> Dict:
    cluster_to_prototype = {}
    for node in aida_graph.nodes('SameAsCluster'):
        prototype = next(iter(node.get('prototype')), None)
        if prototype:
            cluster_to_prototype[node.name] = prototype

    ere_to_prototypes = defaultdict(set)
    for node in aida_graph.nodes('ClusterMembership'):
        cluster_member = next(iter(node.get('clusterMember')), None)
        cluster = next(iter(node.get('cluster')), None)
        if cluster and cluster_member:
            prototype = cluster_to_prototype.get(cluster, None)
            if prototype:
                ere_to_prototypes[cluster_member].add(prototype)

    return ere_to_prototypes


def main():
    parser = ArgumentParser(
        description='Read in a TA2 KB and a (list of) XML-based Statement of Information Need '
                    'definition, convert the KB to JSON format, then convert each SoIN to a JSON '
                    'query by identifying and ranking entry points.')
    parser.add_argument('kb_path', help='Path to the input TA2 KB')
    parser.add_argument('graph_output_path', help='Path to write the JSON graph')
    parser.add_argument('-s', '--soin_path',
                        help='Path to the input SoIN file, or a directory containing multiple SoIN '
                             'files; if not provided, will only transform the graph')
    parser.add_argument('-q', '--query_output_dir',
                        help='Directory to write the JSON queries, used when soin_path is provided')
    parser.add_argument('-m', '--max_matches', type=int, default=50,
                        help='The maximum number of EPs *per entry point description*')
    parser.add_argument('-d', '--dup_kb', default=duplicate_kb_file,
                        help='Path to the json file with duplicate KB ID mappings')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    kb_path = util.get_input_path(args.kb_path)
    graph_output_path = util.get_output_path(
        args.graph_output_path, overwrite_warning=not args.force)

    aida_graph = AidaGraph()
    aida_graph.build_graph(str(kb_path), fmt='ttl')

    json_graph = JsonGraph()
    json_graph.build_graph(aida_graph)

    logging.info('Writing JSON graph to {} ...'.format(graph_output_path))
    with open(str(graph_output_path), 'w') as fout:
        json.dump(json_graph.as_dict(), fout, indent=1)
    logging.info('Done.')

    if args.soin_path is not None:
        assert args.query_output_dir is not None, 'Must provide query_output_dir'
        soin_path = util.get_input_path(args.soin_path)
        query_output_dir = util.get_output_dir(args.query_output_dir,
                                               overwrite_warning=not args.force)

        soin_file_paths = util.get_file_list(soin_path, suffix='.xml', sort=True)

        dup_kb_id_mapping = None
        if args.dup_kb is not None:
            dup_kb_id_mapping = util.read_json_file(args.dup_kb, 'duplicate KB ID mapping')

        logging.info('Getting Cluster Mappings ...')
        ere_to_prototypes = get_cluster_mappings(aida_graph)

        for soin_file_path in soin_file_paths:
            query_output_path = query_output_dir / (soin_file_path.stem + '_query.json')

            logging.info('Processing SOIN {} ...'.format(soin_file_path))
            logging.info('Parsing SOIN XML ...')
            soin = SOIN.parse(str(soin_file_path), dup_kbid_mapping=dup_kb_id_mapping)

            logging.info('Resolving all entrypoints ...')
            soin.resolve(aida_graph, ere_to_prototypes, max_matches=args.max_matches)

            logging.info('Serializing data structures ...')
            query_json = {'graph': kb_path.stem}
            query_json.update(soin.to_json())

            logging.info('Writing JSON query to {} ...'.format(query_output_path))
            with open(str(query_output_path), 'w') as fout:
                json.dump(query_json, fout, indent=1)
            logging.info('Done.')


if __name__ == '__main__':
    main()
