"""
Author: Pengxiang Cheng, Aug 2020
- Adapted from the legacy soin_processing package by Eric.
- Functions to resolve entry points in a statement of information need.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from aida_utexas.aif import AidaGraph
from aida_utexas.soin.entry_point import EntryPoint
from aida_utexas.soin.soin import SOIN

match_score_weight = {'type': 1, 'descriptor': 10}


def get_cluster_mappings(graph: AidaGraph):
    cluster_to_prototype = {}
    entity_to_clusters = defaultdict(set)

    for node in graph.nodes():
        if node.is_same_as_cluster():
            prototype = next(iter(node.get('prototype')), None)
            if prototype:
                cluster_to_prototype[node.name] = prototype
        elif node.is_cluster_membership():
            cluster_member = next(iter(node.get('clusterMember')), None)
            cluster = next(iter(node.get('cluster')), None)
            if cluster and cluster_member:
                entity_to_clusters[cluster_member].add(cluster)

    return cluster_to_prototype, entity_to_clusters


def find_entrypoint(graph: AidaGraph, entrypoint: EntryPoint, cluster_to_prototype: Dict,
                    entity_to_clusters: Dict, max_matches: int) -> Tuple[List, List]:
    """
    A function to resolve an entrypoint to the set of entity nodes that satisfy it.
    This function iterates through every node in the graph. If that node is a typing statement,
    it computes a type score (how many matches between the entity type/subtype/subsubtype) and
    a descriptor score (how many complete TypedDescriptor matches) across all TypedDescriptors.

    The function returns the set of nodes mapped to the highest scores.
    :param graph: AidaGraph
    :param entrypoint: Entrypoint
    :param cluster_to_prototype: dict
    :param entity_to_clusters: dict
    :param max_matches: int
    """
    results = {}

    for node in graph.nodes():
        if node.is_type_statement():
            subj_label = next(iter(node.get('subject', shorten=False)))
            obj_label = next(iter(node.get('object', shorten=True)))

            all_scores = []

            for typed_descriptor in entrypoint.typed_descriptors:
                type_score = 0
                descriptor_score = 0

                has_type = 0
                has_descriptor = 0

                if typed_descriptor.enttype:
                    has_type = 1
                    type_score = typed_descriptor.enttype.match_score(obj_label)

                if typed_descriptor.descriptor:
                    has_descriptor = 1
                    descriptor_score = typed_descriptor.descriptor.match_score(subj_label, graph)

                    if typed_descriptor.descriptor.descriptor_type in ['Text', 'Image', 'Video']:
                        descriptor_score = max(
                            descriptor_score,
                            typed_descriptor.descriptor.match_score(node.name, graph))

                total_score = denominator = 0
                if has_type:
                    total_score += type_score * match_score_weight['type']
                    denominator += match_score_weight['type']
                if has_descriptor:
                    total_score += descriptor_score * match_score_weight['descriptor']
                    denominator += match_score_weight['descriptor']

                all_scores.append(total_score / denominator)

            avg_score = sum(all_scores) / len(all_scores)

            for cluster in entity_to_clusters[subj_label]:
                prototype = cluster_to_prototype[cluster]
                if prototype in results:
                    results[prototype] = max(results[prototype], avg_score)
                else:
                    results[prototype] = avg_score

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    ep_matches, ep_weights = zip(*sorted_results[:max_matches])

    return ep_matches, ep_weights


def resolve_all_entrypoints(graph: AidaGraph, soin: SOIN, cluster_to_prototype: Dict,
                            entity_to_clusters: Dict, max_matches: int = 50):
    ep_matches_dict = {}
    ep_weights_dict = {}
    for entrypoint in soin.entrypoints:
        ep_matches, ep_weights = find_entrypoint(
            graph, entrypoint, cluster_to_prototype, entity_to_clusters, max_matches)
        ep_matches_dict[entrypoint.node] = ep_matches
        ep_weights_dict[entrypoint.node] = ep_weights

    return ep_matches_dict, ep_weights_dict
