"""
    This is a program to process Statements of Information Need (SOINs; provided by DARPA in
    XML), extract the relevant
    query specifications, resolve and rank entrypoints, and produce JSON output structures for
    use downstream by the
    hypothesis creation program.

    The program receives SOIN XML input and outputs a single JSON file.

    TODO:
     - Add video descriptor coverage
     - Replace variable identities
     - Handle cases where resolved EPs are not coreferential

    Author: Eric Holgate
            holgate@utexas.edu
    Update: Pengxiang Cheng, May 2020, for dockerization
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from aida_utexas import util
from aida_utexas.aif import AidaGraph
from aida_utexas.legacy.soin_processing import SOIN
from aida_utexas.legacy.soin_processing.templates_and_constants import DEBUG, SCORE_WEIGHTS, \
    DEBUG_SCORE_FLOOR, ROLE_PENALTY_DEBUG

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')


def get_cluster_mappings(graph):
    cluster_to_prototype = {}
    entities_to_clusters = defaultdict(set)
    entities_to_roles = {}

    for node in graph.nodes():
        if node.is_same_as_cluster():
            cluster_to_prototype[node.name] = next(iter(node.get('prototype')))
        elif node.is_cluster_membership():
            cluster_member = next(iter(node.get('clusterMember')))
            cluster = next(iter(node.get('cluster')))
            entities_to_clusters[cluster_member].add(cluster)
        elif node.is_statement():
            pred_set = node.get('predicate', shorten=True)
            if not pred_set:
                continue
            pred = next(iter(pred_set)).strip()
            if pred != 'type':
                obj_set = node.get('object')
                if not obj_set:
                    continue
                obj = next(iter(obj_set))
                if obj in entities_to_roles:
                    entities_to_roles[obj].add(pred)
                else:
                    entities_to_roles[obj] = {pred}

    return cluster_to_prototype, entities_to_clusters, entities_to_roles


def check_type(node, typed_descriptor):
    """
    A function which determines the extent to which a given AidaGraph node and Entrypoint
    definition contain matching
    type statements.

    :param node: AidaNode
    :param typed_descriptor: TypedDescriptor
    :return: int
    """
    types = next(iter(node.get('object', shorten=True))).strip().split('.')

    # Fix types to length 3 (in case subtype or subsubtype information was missing)
    for i in range(3 - len(types)):
        types.append("")

    types_dict = {
        'type': types[0],
        'subtype': types[1],
        'subsubtype': types[2],
    }

    return typed_descriptor.enttype.get_type_score(types_dict)


def get_subject_node(graph, typing_statement):
    """
    A function to return the subject node for a typing statement node.
    :param graph:
    :param typing_statement:
    :return: AidaNode or False
    """
    subject_node_id_set = typing_statement.get('subject')
    if not subject_node_id_set:
        return False

    subject_node = graph.get_node(next(iter(subject_node_id_set)))
    if not subject_node:
        return False

    return subject_node


def get_justification_node(graph, typing_statement):
    """
    A function to return the justification node for a typing statement node.
    :param graph:
    :param typing_statement:
    :return: AidaNode or False
    """
    justification_node_id_set = typing_statement.get('justifiedBy')
    if not justification_node_id_set:
        return False

    justification_node = graph.get_node(next(iter(justification_node_id_set)))
    if not justification_node:
        return False

    return justification_node


def get_kb_link_node(graph, typing_statement):
    """
    A function to return the KB linking node from a typing statement
    :param graph: AidaGraph
    :param typing_statement: AidaNode
    :return: AidaNode/False
    """
    subject_node = get_subject_node(graph, typing_statement)
    if not subject_node:
        return False
    link_id_set = subject_node.get('link')
    return [graph.get_node(link_id) for link_id in link_id_set]


def get_bounding_box_node(graph, justification_node):
    """
    A function to return the bounding box node from a justification node.
    :param graph: AidaGraph
    :param justification_node: AidaNode
    :return: AidaNode
    """
    bounding_box_id_set = justification_node.get('boundingBox')
    if not bounding_box_id_set:
        return False
    # print(bounding_box_id_set)
    bounding_box_node = graph.get_node(next(iter(bounding_box_id_set)))
    if not bounding_box_node:
        return False
    return bounding_box_node


def check_descriptor(graph, typing_statement, typed_descriptor):
    if typed_descriptor.descriptor.descriptor_type == "Text":
        justification_node = get_justification_node(graph, typing_statement)
        if not justification_node:
            return False
        jtype_set = justification_node.get('type', shorten=True)
        if not jtype_set:
            return False
        jtype = next(iter(jtype_set))
        if jtype != "TextJustification":
            return False

        return typed_descriptor.descriptor.evaluate_node(justification_node)

    elif typed_descriptor.descriptor.descriptor_type == "String":
        subject_node = get_subject_node(graph, typing_statement)
        if not subject_node:
            return False
        return typed_descriptor.descriptor.evaluate_node(subject_node)

    elif typed_descriptor.descriptor.descriptor_type == "Image":
        justification_node = get_justification_node(graph, typing_statement)
        if not justification_node:
            return False
        jtype_set = justification_node.get('type', shorten=True)
        if not jtype_set:
            return False
        jtype = next(iter(jtype_set))
        if jtype != "ImageJustification":
            return False

        bounding_box_node = get_bounding_box_node(graph, justification_node)
        if not bounding_box_node:
            return False
        if not (justification_node and bounding_box_node):
            return False

        return typed_descriptor.descriptor.evaluate_node(justification_node, bounding_box_node)

    elif typed_descriptor.descriptor.descriptor_type == "Video":
        justification_node = get_justification_node(graph, typing_statement)
        if not justification_node:
            return False

        jtype_set = justification_node.get('type', shorten=True)
        if not jtype_set:
            return False

        jtype = next(iter(jtype_set))
        if jtype != "KeyFrameVideoJustification":
            return False

        bounding_box_node = get_bounding_box_node(graph, justification_node)
        if not bounding_box_node:
            return False

        return typed_descriptor.descriptor.evaluate_node(justification_node, bounding_box_node)

    elif typed_descriptor.descriptor.descriptor_type == "KB":
        link_node_scores = []
        for link_node in get_kb_link_node(graph, typing_statement):
            link_node_scores.append(typed_descriptor.descriptor.evaluate_node(link_node))

        if not link_node_scores:
            return False
        else:
            return max(link_node_scores)

    return False


def find_entrypoint(graph, entrypoint, cluster_to_prototype, entity_to_cluster, entities_to_roles,
                    role_vars, ep_cap, role_flag):
    """
    A function to resolve an entrypoint to the set of entity nodes that satisfy it.
    This function iterates through every node in the graph. If that node is a typing statement,
    it computes a
    typed score (how many matches between enttypes) and descriptor score (how many complete
    TypedDescriptor matches)
    across all TypedDescriptors. These scores are mapped typed_score -> descriptor_score -> {Nodes}.

    The function returns the set of nodes mapped to the highest scores (i.e.,
     highest typed_score -> highest descriptor_score).
    :param graph: AidaGraph
    :param entrypoint: Entrypoint
    :param cluster_to_prototype: dict
    :param entity_to_cluster: dict
    :param ep_cap: int
    :return: {Nodes}
    """
    results = {}
    for node in graph.nodes():

        if node.is_type_statement():
            typed_score = 0
            name_score = 0
            descriptor_score = 0

            has_type = 0
            has_name = 0
            has_descriptor = 0

            num_enttypes = 0
            num_descriptors = 0

            # TODO: Why is this wrapped in a useless tuple??
            for filler in entrypoint.typed_descriptor_list:
                for typed_descriptor in filler:
                    if typed_descriptor.enttype:
                        has_type = 1
                        num_enttypes += 1
                        typed_score += check_type(node, typed_descriptor)
                    if typed_descriptor.descriptor:
                        if typed_descriptor.descriptor.descriptor_type == 'String':
                            has_name = 1
                            name_score += check_descriptor(graph, node, typed_descriptor)
                        else:
                            has_descriptor = 1
                            num_descriptors += 1
                            descriptor_score += check_descriptor(graph, node, typed_descriptor)

            subject_address = next(iter(node.get('subject')))
            try:
                prototypes = [cluster_to_prototype[cluster] for cluster in
                              entity_to_cluster[subject_address]]
            except KeyError:
                if DEBUG:
                    print("KEY ERROR IN PROTOTYPE MAPPINGS!!")
                continue

            # Compute the total score, pull the prototype, and add the prototype node to the
            # results dict
            # Compute the denominator for the score based on what information was present
            raw_score = (typed_score / 100) + (name_score / 100) + (descriptor_score / 100)
            score_numerator = 0
            score_denominator = 0

            if has_type:
                score_numerator += ((typed_score / num_enttypes) / 100) * SCORE_WEIGHTS['type']
                score_denominator += SCORE_WEIGHTS['type']
            if has_name:
                score_numerator += (name_score / 100) * SCORE_WEIGHTS['name']
                score_denominator += SCORE_WEIGHTS['name']
            if has_descriptor:
                score_numerator += ((descriptor_score / num_descriptors) / 100) * SCORE_WEIGHTS[
                    'descriptor']
                score_denominator += SCORE_WEIGHTS['descriptor']

            total_score = (score_numerator / score_denominator) * 100
            if role_flag:
                penalty = 0
                for role in role_vars[entrypoint.variable[0]]:  # [0] for senseless tuple wrapper
                    if role not in entities_to_roles.get(subject_address, {}):
                        if ROLE_PENALTY_DEBUG:
                            print("PENALTY APPLIED!")
                            print("Looking for: " + str(role))
                            print("Observed roles: " + str(
                                entities_to_roles.get(subject_address, {})))
                            input()
                        penalty += 30
                total_score = total_score - penalty

            if DEBUG:
                print("Raw Score: " + str(raw_score))
                print("Score Numerator: " + str(score_numerator))
                print("Score Denominator: " + str(score_denominator))

            if DEBUG:
                print("Normalized Score: " + str(total_score))
                print()
                print("##############################################")
                print()
                if (total_score >= DEBUG_SCORE_FLOOR):
                    input()

            for prototype in prototypes:
                if prototype in results:
                    prev_score = results[prototype]
                    if total_score > prev_score:
                        results[prototype] = total_score
                else:
                    results[prototype] = total_score
            # if total_score in results:
            #     results[total_score].update([(total_score, prototype) for prototype in
            #     prototypes])
            # else:
            #     results[total_score] = set([(total_score, prototype) for prototype in prototypes])

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    ep_list, ep_weight_list = zip(*sorted_results[:ep_cap])

    # set_of_nodes = set()
    # return_count = 0
    # scores_sorted = sorted(results.keys(), reverse=True)
    # for score in scores_sorted:
    #     for node in results[score]:
    #         if return_count >= ep_cap:
    #             break
    #         elif node not in set_of_nodes:
    #             return_count += 1
    #             set_of_nodes.add(node)

    # ordered_list = sorted(list(set_of_nodes), key=lambda x: x[0], reverse=True)
    # ep_list = []
    # ep_weight_list = []
    # for elem in ordered_list:
    #     ep_list.append(elem[1])
    #     ep_weight_list.append(elem[0])
    return ep_list, ep_weight_list


def resolve_all_entrypoints(graph, entrypoints, cluster_to_prototype, entity_to_cluster,
                            entities_to_roles, var_roles, ep_cap, roles_flag):
    ep_dict = {}
    ep_weight_dict = {}
    for entrypoint in entrypoints:
        # results[entrypoint.variable[0]]
        ep_list, ep_weight_list = find_entrypoint(graph,
                                                  entrypoint,
                                                  cluster_to_prototype,
                                                  entity_to_cluster,
                                                  entities_to_roles,
                                                  var_roles,
                                                  ep_cap,
                                                  roles_flag)
        ep_dict[entrypoint.variable[0]] = ep_list
        ep_weight_dict[entrypoint.variable[0]] = ep_weight_list

    return ep_dict, ep_weight_dict


def process_soin(graph: AidaGraph, soin_file_paths: List[Path], output_dir: Path, ep_cap: int = 50,
                 consider_roles: bool = False, dup_kb_id_mapping: Dict = None):
    logging.info("Getting Cluster Mappings ...")
    cluster_to_prototype, entity_to_cluster, entities_to_roles = get_cluster_mappings(graph)

    for soin_file_path in soin_file_paths:
        logging.info('Processing SOIN {} ...'.format(soin_file_path))
        logging.info('Parsing SOIN XML ...')
        soin = SOIN.process_xml(str(soin_file_path), dup_kbid_mapping=dup_kb_id_mapping)
        # logging.info('Done.')

        logging.info('Gathering role information ...')
        ep_variables = set()
        for ep in soin.entrypoints:
            ep_variables.add(ep.variable[0])  # [0] is for senseless tuple wrapper

        var_roles = {}
        for frame in soin.frames:
            for edge in frame.edge_list:
                if edge.obj in ep_variables:
                    if edge.obj in var_roles:
                        var_roles[edge.obj].add(edge.predicate)
                    else:
                        var_roles[edge.obj] = {edge.predicate}
        # logging.info('Done.')

        logging.info('Resolving all entrypoints ...')
        ep_dict, ep_weights_dict = resolve_all_entrypoints(graph,
                                                           soin.entrypoints,
                                                           cluster_to_prototype,
                                                           entity_to_cluster,
                                                           entities_to_roles,
                                                           var_roles,
                                                           ep_cap,
                                                           consider_roles)

        # logging.info('Done.')

        write_me = {
            'graph': '',
            'soin_id': soin.id,
            'frame_id': [frame.id for frame in soin.frames],
            'entrypoints': ep_dict,
            'entrypointWeights': ep_weights_dict,
            'queries': [],
            'facets': [],
        }

        logging.info('Serializing data structures ...')
        temporal_info = soin.temporal_info_to_dict()
        for frame in soin.frames:
            frame_rep = frame.frame_to_dict(temporal_info)
            write_me['facets'].append(frame_rep)

        query_output = util.get_output_path(output_dir / (soin_file_path.stem + '_query.json'))
        logging.info('Writing JSON query to {} ...'.format(query_output))
        with open(str(query_output), 'w') as fout:
            json.dump(write_me, fout, indent=1)
        logging.info('Done.')
