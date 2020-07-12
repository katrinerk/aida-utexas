import json
from argparse import ArgumentParser
from collections import defaultdict

import pandas

from aida_utexas import util


def main():
    parser = ArgumentParser()
    parser.add_argument('input_path', help='path to the input Excel ontology file')
    parser.add_argument('output_path', help='path to write the JSON ontology file')

    args = parser.parse_args()

    input_path = util.get_input_path(args.input_path)

    df = pandas.read_excel(str(input_path), sheet_name=None)
    event_records = df['events'].to_dict('records')
    relation_records = df['relations'].to_dict('records')

    roles_ontology = defaultdict(dict)

    for ev in event_records:
        ev_type = ev['Type']
        if isinstance(ev['Subtype'], str):
            ev_type += '.' + ev['Subtype']
            if isinstance(ev['Sub-subtype'], str):
                ev_type += '.' + ev['Sub-subtype']

        for arg_idx in range(1, 6):
            arg_key = f'arg{arg_idx} label'
            if isinstance(ev[arg_key], str):
                roles_ontology[ev_type][f'arg{arg_idx}'] = ev[arg_key]

    for rel in relation_records:
        rel_type = rel['Type']
        if isinstance(rel['Subtype'], str):
            rel_type += '.' + rel['Subtype']
            if isinstance(rel['Sub-Subtype'], str):
                rel_type += '.' + rel['Sub-Subtype']

        roles_ontology[rel_type]['arg1'] = rel['arg1 label']
        roles_ontology[rel_type]['arg2'] = rel['arg2 label']

    output_path = util.get_output_path(args.output_path)
    with open(str(output_path), 'w') as fout:
        json.dump(roles_ontology, fout, indent=2)


if __name__ == '__main__':
    main()
