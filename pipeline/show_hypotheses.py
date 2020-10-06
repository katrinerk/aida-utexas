from argparse import ArgumentParser

from aida_utexas import util
from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis.aida_hypothesis import AidaHypothesisCollection


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_path', help='path to the graph JSON file')
    parser.add_argument('hypothesis_path', help='path to the JSON file with hypotheses')
    parser.add_argument('roles_ontology_path', help='path to the roles ontology file')
    parser.add_argument('output_dir', help='directory to write human-readable hypotheses')

    args = parser.parse_args()

    json_graph = JsonGraph.from_dict(util.read_json_file(args.graph_path, 'JSON graph'))

    hypotheses_json = util.read_json_file(args.hypothesis_path, 'hypotheses')
    hypothesis_collection = AidaHypothesisCollection.from_json(hypotheses_json, json_graph)

    roles_ontology = util.read_json_file(args.roles_ontology_path, 'roles ontology')

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=True)

    for idx, hypothesis in enumerate(hypothesis_collection.hypotheses):
        output_path = output_dir / 'hypothesis-{:0>3d}.txt'.format(idx)
        with open(str(output_path), "w") as fout:
            print(hypothesis.to_str(roles_ontology), file=fout)


if __name__ == '__main__':
    main()
