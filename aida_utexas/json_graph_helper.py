"""
Author: Pengxiang Cheng, Aug 2019
- Helper function to construct mappings on clusters, members, and prototypes from an AidaGraph.

Update: Pengxiang Cheng, Aug 2020
- Adapt to new JsonGraph APIs.
"""

from collections import Counter, defaultdict

from aida_utexas.aif import JsonGraph


def build_cluster_member_mappings(json_graph: JsonGraph, debug=False):
    # Build mappings between clusters and members, and mappings between
    # clusters and prototypes
    print('\nBuilding mappings among clusters, members and prototypes ...')

    cluster_to_members = defaultdict(set)
    member_to_clusters = defaultdict(set)
    cluster_membership_key_mapping = defaultdict(set)

    cluster_to_prototype = {}
    prototype_to_clusters = defaultdict(set)

    for node_label, node in json_graph.node_dict.items():
        if node.type == 'ClusterMembership':
            cluster, member = node.cluster, node.clusterMember
            assert cluster is not None and member is not None

            cluster_to_members[cluster].add(member)
            member_to_clusters[member].add(cluster)

            if debug and (cluster, member) in cluster_membership_key_mapping:
                print('Warning: found duplicate ClusterMembership nodes for ({}, {})'.format(
                    cluster, member))
            cluster_membership_key_mapping[(cluster, member)].add(node_label)

        elif node.type == 'SameAsCluster':
            assert node_label not in cluster_to_prototype
            assert node.prototype is not None

            cluster_to_prototype[node_label] = node.prototype
            prototype_to_clusters[node.prototype].add(node_label)

    num_clusters = len(cluster_to_members)
    num_members = len(member_to_clusters)
    num_prototypes = len(prototype_to_clusters)
    assert len(cluster_to_prototype) == num_clusters

    print('\nConstructed mapping from {} clusters to {} members'.format(
        num_clusters, num_members))

    clusters_per_member_counter = Counter([len(v) for v in member_to_clusters.values()])
    for key in sorted(clusters_per_member_counter.keys()):
        if key > 1:
            print(
                '\tFor {} out of {} members, each belong to {} clusters'.format(
                    clusters_per_member_counter[key], num_members, key))

    print('\nConstructed mapping from {} cluster-member pairs to '
          'their ClusterMembership node labels'.format(len(cluster_membership_key_mapping)))

    cm_node_per_pair_counter = Counter(len(v) for v in cluster_membership_key_mapping.values())
    for key in sorted(cm_node_per_pair_counter.keys()):
        if key > 1:
            print(
                '\tFor {} out of {} cluster-member pairs, each is associated with '
                '{} ClusterMembership nodes'.format(
                    cm_node_per_pair_counter[key], len(cluster_membership_key_mapping), key))

    print('\nConstructed mapping from {} clusters to {} prototypes'.format(
        num_clusters, num_prototypes))

    clusters_per_prototype_counter = Counter([len(v) for v in prototype_to_clusters.values()])
    for key in sorted(clusters_per_prototype_counter.keys()):
        if key > 1:
            print(
                '\tFor {} out of {} prototypes, each is the prototype of {} '
                'clusters'.format(
                    clusters_per_prototype_counter[key], num_prototypes, key))

    # Build mappings between members and prototypes, using the above
    # constructed mappings
    member_to_prototypes = defaultdict(set)
    prototype_to_members = defaultdict(set)

    for member, clusters in member_to_clusters.items():
        assert member not in member_to_prototypes
        for cluster in clusters:
            prototype = cluster_to_prototype[cluster]
            member_to_prototypes[member].add(prototype)
            prototype_to_members[prototype].add(member)

    assert len(member_to_prototypes) == num_members
    assert len(prototype_to_members) == num_prototypes

    print('\nConstructed mapping from {} members to {} prototypes'.format(
        num_members, num_prototypes))

    prototypes_per_member_counter = Counter([len(v) for v in member_to_prototypes.values()])
    for key in sorted(prototypes_per_member_counter.keys()):
        if key > 1:
            print(
                '\tFor {} out of {} members, each is mapped to {} '
                'prototypes'.format(
                    prototypes_per_member_counter[key], num_members, key))

    # Add ERE nodes that are not connected to any ClusterMembership node to
    # the mappings between members and prototypes. This shouldn't happen,
    # unless the TA2 output we get don't conform to the NIST-restricted
    # formatting requirements.
    ere_nodes_not_in_clusters = set()
    for node_label, node in json_graph.node_dict.items():
        if node.type in ['Entity', 'Relation', 'Event']:
            if node_label not in member_to_prototypes:
                ere_nodes_not_in_clusters.add(node_label)
    if len(ere_nodes_not_in_clusters) > 0:
        print('\nWarning: Found {} ERE nodes that are not connected to any '
              'ClusterMembership node'.format(len(ere_nodes_not_in_clusters)))
        print('Adding them to the mappings between members and prototypes')
        for node_label in ere_nodes_not_in_clusters:
            member_to_prototypes[node_label].add(node_label)
            prototype_to_members[node_label].add(node_label)
        print(
            '\nAfter correction, constructed mapping from {} members to '
            '{} prototypes'.format(
                len(member_to_prototypes), len(prototype_to_members)))

    mappings = {
        'cluster_to_members': cluster_to_members,
        'member_to_clusters': member_to_clusters,
        'cluster_membership_key_mapping': cluster_membership_key_mapping,
        'cluster_to_prototype': cluster_to_prototype,
        'prototype_to_clusters': prototype_to_clusters,
        'member_to_prototypes': member_to_prototypes,
        'prototype_to_members': prototype_to_members,
    }

    return mappings
