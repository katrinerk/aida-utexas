from argparse import ArgumentParser
from operator import itemgetter

from aida_utexas import util
from aida_utexas.aif import JsonGraph, AIDA, LDC, LDC_ONT, UTEXAS

update_prefix = \
    f'PREFIX ldcOnt: {LDC_ONT}\n' \
    f'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n' \
    f'PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n' \
    f'PREFIX aida: {AIDA}\n' \
    f'PREFIX ldc: {LDC}\n' \
    f'PREFIX utexas: {UTEXAS}\n\n'


def compute_handle_mapping(json_graph, hypothesis, member_to_clusters, cluster_to_prototype):
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

    prototype_handles = {}
    for cluster in cluster_set:
        prototype = cluster_to_prototype.get(cluster, None)
        if prototype is not None:
            prototype_handles[prototype] = json_graph.node_dict[cluster].handle

    return prototype_handles


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_path', help='path to the graph json file')
    parser.add_argument('hypotheses_path', help='path to the hypotheses json file')
    parser.add_argument('output_dir', help='Directory to write queries')
    parser.add_argument('--top', default=50, type=int,
                        help='number of top hypothesis to output')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    json_graph = JsonGraph.from_dict(util.read_json_file(args.graph_path, 'JSON graph'))
    mappings = json_graph.build_cluster_member_mappings()

    hypotheses_json = util.read_json_file(args.hypotheses_path, 'hypotheses')

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=not args.force)

    top_count = 0

    for result_idx, prob in sorted(
            enumerate(hypotheses_json['probs']), key=itemgetter(1), reverse=True):
        hypothesis = hypotheses_json['support'][result_idx]

        top_count += 1

        update_str = update_prefix + 'INSERT DATA\n{\n'

        prototype_handles = compute_handle_mapping(
            json_graph, hypothesis, member_to_clusters=mappings['member_to_clusters'],
            cluster_to_prototype=mappings['cluster_to_prototype'])

        for prototype, handle in prototype_handles.items():
            handle = handle.lstrip('"')
            handle = handle.rstrip('"')
            update_str += '  <{}> aida:handle "{}" .\n'.format(prototype, handle)

        update_str += '}'

        output_path = output_dir / 'hypothesis-{:0>3d}-update.rq'.format(top_count)

        with open(str(output_path), 'w') as fout:
            fout.write(update_str)

        if top_count >= args.top:
            break


if __name__ == '__main__':
    main()
