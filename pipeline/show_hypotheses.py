import json
from argparse import ArgumentParser

from aida_utexas import util
from aida_utexas.aif import AidaJson
from aida_utexas.seeds.aidahypothesis import AidaHypothesisCollection


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_path', help='path to the graph JSON file')
    parser.add_argument('hypothesis_path', help='path to the JSON file with hypotheses')
    parser.add_argument('roles_ontology_path', help='path to the roles ontology file')
    parser.add_argument('output_dir', help='directory to write human-readable hypotheses')

    args = parser.parse_args()

    graph_path = util.get_input_path(args.graph_path)
    with open(str(graph_path), 'r') as fin:
        graph_json = AidaJson(json.load(fin))

    hypothesis_path = util.get_input_path(args.hypothesis_path)
    with open(str(hypothesis_path), 'r') as fin:
        json_hypotheses = json.load(fin)
        hypothesis_collection = AidaHypothesisCollection.from_json(json_hypotheses, graph_json)

    roles_ontology_path = util.get_input_path(args.roles_ontology_path)
    with open(str(roles_ontology_path), 'r') as fin:
        roles_ontology = json.load(fin)

    output_dir = util.get_dir(args.output_dir, create=True)

    for idx, hypothesis in enumerate(hypothesis_collection.hypotheses):
        output_path = output_dir / 'hypothesis-{:0>3d}.txt'.format(idx)
        with open(str(output_path), "w") as fout:
            print(hypothesis.to_s(roles_ontology), file=fout)


if __name__ == '__main__':
    main()
