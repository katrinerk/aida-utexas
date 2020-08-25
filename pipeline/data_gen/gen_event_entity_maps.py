import dill
import argparse
from gen_salads import Indexer

# Given a directory containing single-doc Wiki json files, this script generates event and entity name dicts for use in creating
# graph salads.
# e.g., event_name_dict['France'] = {'graph_1_ere_2', 'graph_5_ere_7', ...}

def process_ere_list(graph, ere_list, names_dict, types_dict, conn_one_dict, conn_two_dict):
    for (ere_id, ere) in ere_list:
        if len([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if (graph.stmts[stmt_id].tail_id and (graph.eres[graph.stmts[stmt_id].head_id].category == 'Event'))]) < 1:
            continue

        type_stmts = set([graph.stmts[stmt_id].raw_label for stmt_id in ere.stmt_ids if not graph.stmts[stmt_id].tail_id and graph.stmts[stmt_id].head_id == ere_id])

        [names_dict[label].add(ere_id) for label in ere.label]

        types_dict[ere_id] = type_stmts
        conn_one_dict[ere_id] = graph.connectedness_one_step[ere_id]
        conn_two_dict[ere_id] = graph.connectedness_two_step[ere_id]

# Simple heuristic for accounting (roughly) for differences in capitalization and/or plurality
def simple_merge(names_dict):
    checked = set()

    while len(checked) < len(names_dict.keys()):
        key_1 = list(set(names_dict.keys()) - checked)[0]
        checked.add(key_1)

        for key_2 in (set(names_dict.keys()) - checked):
            if (key_1.lower() == key_2.lower() or key_1.lower() == key_2.lower() + 's' or key_1.lower() + 's' == key_2.lower()):
                names_dict[key_1] = set.union(names_dict[key_1], names_dict[key_2])
                del names_dict[key_2]

def generate_name_lists(graph_dir, output_dir):
    event_names = defaultdict(set)
    entity_names = defaultdict(set)

    event_types = dict()
    entity_types = dict()

    connectedness_one_step = dict()
    connectedness_two_step = dict()

    for file_name in os.listdir(graph_dir):
        graph = dill.load(open(os.path.join(graph_dir, file_name), 'rb'))

        events = [(ere_id, ere) for (ere_id, ere) in graph.eres.items() if ere.category == 'Event']
        entities = [(ere_id, ere) for (ere_id, ere) in graph.eres.items() if ere.category == 'Entity']

        process_ere_list(graph, events, event_names, event_types, connectedness_one_step, connectedness_two_step)
        process_ere_list(graph, entities, entity_names, entity_types, connectedness_one_step, connectedness_two_step)

    simple_merge(event_names)
    simple_merge(entity_names)

    dill.dump(event_names, open(os.path.join(output_dir, 'event_names.p'), 'wb'))
    dill.dump(entity_names, open(os.path.join(output_dir, 'entity_names.p'), 'wb'))
    dill.dump(event_types, open(os.path.join(output_dir, 'event_types.p'), 'wb'))
    dill.dump(entity_types, open(os.path.join(output_dir, 'entity_types.p'), 'wb'))
    dill.dump(connectedness_one_step, open(os.path.join(output_dir, 'connectedness_one_step.p'), 'wb'))
    dill.dump(connectedness_two_step, open(os.path.join(output_dir, 'connectedness_two_step.p'), 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="Single_Doc_Graphs/", help='Directory containing all single-doc graph instances (as pickled objects)')
    parser.add_argument("--output_dir", type=str, default="", help='Directory where output will be written')

    args = parser.parse_args()

    generate_name_lists(args.graph_dir, args.output_dir)



