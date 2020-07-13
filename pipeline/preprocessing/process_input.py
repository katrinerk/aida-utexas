"""
Author: Katrin Erk October 2018
generates a json file called aidagraph.json that describes an AIDA graph, the units for
clustering, and baseline values for cluster distances.

Update: Pengxiang Cheng August 2019
rewrite for M18 evaluation

Update: Pengxiang Cheng May 2020
incorporate the SoIN processing script into this file to avoid loading the same TTL twice.
"""

import json
import logging
from argparse import ArgumentParser

import rdflib

from aida_utexas import util
from aida_utexas.aif import AidaGraph, JsonInterface
from aida_utexas.soin_processing import process_soin

duplicate_kb_file = 'resources/duplicate_kb_id_mapping.json'


def read_graph(kb_path: str):
    graph = AidaGraph()

    logging.info('Loading RDF graph from {} ...'.format(kb_path))
    rdf_graph = rdflib.Graph()
    rdf_graph.parse(kb_path, format='ttl')
    logging.info('Done with {} triples.'.format(len(rdf_graph)))

    logging.info('Building AidaGraph ...')
    graph.add_graph(rdf_graph)
    logging.info('Done.')

    return graph


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
    parser.add_argument('-c', '--ep_cap', type=int, default=50,
                        help='The maximum number of EPs *per entry point description*')
    parser.add_argument('-r', '--roles', action='store_true', default=False,
                        help='This flag tells the program to consider role information')
    parser.add_argument('-d', '--dup_kb', default=duplicate_kb_file,
                        help='Path to the json file with duplicate KB ID mappings')

    args = parser.parse_args()

    kb_path = util.get_input_path(args.kb_path)
    soin_path = util.get_input_path(args.soin_path)
    graph_output_path = util.get_output_path(args.graph_output_path)
    query_output_dir = util.get_dir(args.query_output_dir, create=True)

    graph = read_graph(str(kb_path))

    logging.info('Building JSON representation of the AIF graph ...')
    graph_json = JsonInterface(graph, simplification_level=0)
    logging.info('Done.')

    logging.info('Writing JSON graph to {} ...'.format(graph_output_path))
    with open(str(graph_output_path), 'w') as fout:
        graph_json.write(fout)
    logging.info('Done.')

    soin_file_paths = util.get_file_list(soin_path, suffix='.xml', sort=True)

    dup_kb_id_mapping = None
    if args.dup_kb is not None:
        logging.info('Loading duplicate KB ID mapping from {} ...'.format(args.dup_kb))
        with open(args.dup_kb, 'r') as fin:
            dup_kb_id_mapping = json.load(fin)

    process_soin(
        graph=graph,
        soin_file_paths=soin_file_paths,
        output_dir=query_output_dir,
        ep_cap=args.ep_cap,
        consider_roles=args.roles,
        dup_kb_id_mapping=dup_kb_id_mapping
    )


if __name__ == '__main__':
    main()
