import json
from argparse import ArgumentParser
from operator import itemgetter

from aida_utexas import util
from aida_utexas.aif import JsonGraph

update_prefix = \
    'PREFIX ldcOnt: <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LDCOntology#>\n' \
    'PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n' \
    'PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>\n' \
    'PREFIX aida:  <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/InterchangeOntology#>\n' \
    'PREFIX ldc:   <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LdcAnnotations#>\n' \
    'PREFIX utexas: <http://www.utexas.edu/aida/>\n\n'


def compute_handle_mapping(json_graph, hypothesis, member_to_clusters):
    cluster_set = set()

    for stmt_label in hypothesis['statements']:
        stmt_subj = json_graph.stmt_subject(stmt_label)
        stmt_obj = json_graph.stmt_object(stmt_label)
        assert stmt_subj is not None and stmt_obj is not None

        if json_graph.is_entity(stmt_subj):
            for cluster in member_to_clusters[stmt_subj]:
                cluster_set.add(cluster)

        if json_graph.is_entity(stmt_obj):
            for cluster in member_to_clusters[stmt_obj]:
                cluster_set.add(cluster)

    cluster_handles = {}
    for cluster in cluster_set:
        cluster_handles[cluster] = json_graph.node_dict[cluster].handle

    return cluster_handles


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_json_path', help='path to the graph json file')
    parser.add_argument('hypotheses_json_path', help='path to the hypotheses json file')
    parser.add_argument('output_dir', help='Directory to write queries')
    parser.add_argument('--top', default=14, type=int,
                        help='number of top hypothesis to output')

    args = parser.parse_args()

    graph_json_path = util.get_input_path(args.graph_json_path)
    hypotheses_json_path = util.get_input_path(args.hypotheses_json_path)
    output_dir = util.get_dir(args.output_dir, create=True)

    json_graph = JsonGraph.load(graph_json_path)

    member_to_clusters = json_graph.build_cluster_member_mappings()['member_to_clusters']

    print('Reading the hypotheses from {}'.format(hypotheses_json_path))
    with open(str(hypotheses_json_path), 'r') as fin:
        hypotheses_json = json.load(fin)

    print('Found {} hypotheses with probabilities of {}'.format(
        len(hypotheses_json['probs']), hypotheses_json['probs']))

    top_count = 0

    for result_idx, prob in sorted(
            enumerate(hypotheses_json['probs']), key=itemgetter(1), reverse=True):
        hypothesis = hypotheses_json['support'][result_idx]

        top_count += 1

        update_str = update_prefix + 'INSERT DATA\n{\n'

        cluster_handles = compute_handle_mapping(json_graph, hypothesis, member_to_clusters)

        for cluster, handle in cluster_handles.items():
            handle = handle.lstrip('"')
            handle = handle.rstrip('"')
            update_str += '  <{}> aida:handle "{}" .\n'.format(cluster, handle)

        update_str += '}'

        output_path = output_dir / 'hypothesis-{:0>3d}-update.rq'.format(top_count)

        with open(str(output_path), 'w') as fout:
            fout.write(update_str)

        if top_count >= args.top:
            break


if __name__ == '__main__':
    main()
