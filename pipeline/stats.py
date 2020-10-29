import argparse
import json
from collections import defaultdict

from aida_utexas import util
from aida_utexas.aif import AidaGraph, JsonGraph


def get_json_stats(json_graph: JsonGraph):
    eres = []
    singleton_eres = []
    entities = []
    singleton_entities = []
    relations = []
    singleton_relations = []
    events = []
    singleton_events = []
    stmts = []
    type_stmts = []
    clusters = []
    cluster_memberships = []
    prototypes = []
    ere_to_memberships = defaultdict(set)
    ere_to_clusters = defaultdict(set)

    for node_label, node in json_graph.node_dict.items():
        if json_graph.is_ere(node_label):
            eres.append(node_label)

            is_singleton = True
            for stmt_label in json_graph.each_ere_adjacent_stmt(node_label):
                if not json_graph.is_type_stmt(stmt_label):
                    is_singleton = False
                    break
            if is_singleton:
                singleton_eres.append(node_label)

            if json_graph.is_entity(node_label):
                entities.append(node_label)
                if is_singleton:
                    singleton_entities.append(node_label)
            if json_graph.is_relation(node_label):
                relations.append(node_label)
                if is_singleton:
                    singleton_relations.append(node_label)
            if json_graph.is_event(node_label):
                events.append(node_label)
                if is_singleton:
                    singleton_events.append(node_label)

        if json_graph.is_statement(node_label):
            stmts.append(node_label)
        if json_graph.is_type_stmt(node_label):
            type_stmts.append(node_label)

        if node.type == 'SameAsCluster':
            clusters.append(node_label)
            prototypes.append(node.prototype)
            ere_to_clusters[node.prototype].add(node_label)

        if node.type == 'ClusterMembership':
            cluster_memberships.append(node_label)
            clusters.append(node.cluster)
            ere_to_clusters[node.clusterMember].add(node.cluster)
            ere_to_memberships[node.clusterMember].add(node_label)

    print(f'# Nodes: {len(json_graph.node_dict)}')
    print(f'# EREs: {len(eres)} ({len(singleton_eres)} are singleton)')
    print(f'# Entities: {len(entities)} ({len(singleton_entities)} are singleton)')
    print(f'# Relations: {len(relations)} ({len(singleton_relations)} are singleton)')
    print(f'# Events: {len(events)} ({len(singleton_events)} are singleton)')
    print(f'# Statements: {len(stmts)}')
    print(f'# Type Statements: {len(type_stmts)}')
    print(f'# SameAsClusters: {len(clusters)}')
    print(f'# ClusterMemberships: {len(cluster_memberships)}')
    print(f'# Prototype EREs: {len(prototypes)}')

    num_clusters_per_ere = [len(val) for val in ere_to_clusters.values()]
    print(f'# Clusters per ERE: min = {min(num_clusters_per_ere)}, '
          f'max = {max(num_clusters_per_ere)}, '
          f'mean = {sum(num_clusters_per_ere) / len(num_clusters_per_ere)}')

    num_memberships_per_ere = [len(val) for val in ere_to_memberships.values()]
    print(f'# Memberships per ERE: min = {min(num_memberships_per_ere)}, '
          f'max = {max(num_memberships_per_ere)}, '
          f'mean = {sum(num_memberships_per_ere) / len(num_memberships_per_ere)}')


def get_kb_singleton_ere(kb_graph, node_type):
    # a singleton ERE is one that only has type statements but no other statements
    singleton_node = []

    for node in kb_graph.nodes(node_type):
        singleton = True

        for pred, subj_labels in node.in_edge.items():
            if not singleton:
                break
            for subj_label in subj_labels:
                subj_node = kb_graph.get_node(subj_label)
                if subj_node and subj_node.is_statement() and not subj_node.is_type_statement():
                    singleton = False
                    break

        if singleton:
            singleton_node.append(node)

    return singleton_node


def get_kb_stats(kb_graph: AidaGraph):
    num_stmts = len(list(kb_graph.nodes('Statement')))
    num_type_stmts = len(list(
        s for s in kb_graph.nodes('Statement') if s.has_predicate("type", shorten=True)))

    num_entities = len(list(kb_graph.nodes('Entity')))
    num_singleton_entities = len(get_kb_singleton_ere(kb_graph, 'Entity'))

    num_relations = len(list(kb_graph.nodes('Relation')))
    num_singleton_relations = len(get_kb_singleton_ere(kb_graph, 'Relation'))

    num_events = len(list(kb_graph.nodes('Event')))
    num_singleton_events = len(get_kb_singleton_ere(kb_graph, 'Event'))

    num_eres = num_entities + num_relations + num_events
    num_singleton_eres = num_singleton_entities + num_singleton_relations + num_singleton_events

    print(f'# Nodes: {len(list(kb_graph.nodes()))}')
    print(f'# EREs: {num_eres} ({num_singleton_eres} are singleton)')
    print(f'# Entities: {num_entities} ({num_singleton_entities} are singleton)')
    print(f'# Relations: {num_relations} ({num_singleton_relations} are singleton)')
    print(f'# Events: {num_events} ({num_singleton_events} are singleton)')
    print(f'# Statements: {num_stmts}')
    print(f'# Type Statements: {num_type_stmts}')
    print(f'# SameAsClusters: {len(list(kb_graph.nodes("SameAsCluster")))}')
    print(f'# ClusterMemberships: {len(list(kb_graph.nodes("ClusterMembership")))}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to a TA2 KB or a JSON graph')

    args = parser.parse_args()

    input_path = util.get_input_path(args.input_path)

    if input_path.suffix == '.ttl':
        aida_graph = AidaGraph()
        aida_graph.build_graph(str(input_path), fmt='ttl')

        get_kb_stats(aida_graph)

    elif input_path.suffix == '.json':
        with open(input_path, 'r') as fin:
            json_graph = JsonGraph.from_dict(json.load(fin))

        get_json_stats(json_graph)


if __name__ == '__main__':
    main()
