"""
Author: Katrin Erk, Oct 2018
- Build on RDFGraph, has higher-level access on AIDA-specific ontologies

Updated: Pengxiang Cheng, Aug 2019
- Update for M18 evaluation

Updated: Pengxiang Cheng, Aug 2020
- Cleanup and refactoring
"""

import json
from collections import deque
from typing import Iterable

from aida_utexas.aif.rdf_graph import RDFGraph, RDFNode

omit_labels = ('system', 'confidence', 'privateData', 'justifiedBy')


# AidaNode: an RDFNode, extended by domain-specific stuff
class AidaNode(RDFNode):
    def __init__(self, node_name):
        RDFNode.__init__(self, node_name)
        self.description = None

    def add_description(self, description):
        self.description = description

    def to_text(self, omit=omit_labels):
        node_str = super().prettyprint(omit=omit)
        if self.description is not None:
            node_str += '\n\tdescr: {}'.format(self.description)
        return node_str

    # if the node has type of target_type
    def has_type(self, target_type, shorten=True):
        return target_type in self.get('type', shorten=shorten)

    # if the node is an Entity
    def is_entity(self):
        return self.has_type('Entity', shorten=True)

    # if the node is a Relation
    def is_relation(self):
        return self.has_type('Relation', shorten=True)

    # if the node is an Event
    def is_event(self):
        return self.has_type('Event', shorten=True)

    # if the node is an Entity / Relation / Event
    def is_ere(self):
        return self.is_entity() or self.is_relation() or self.is_event()

    # if the node is a Statement
    def is_statement(self):
        return self.has_type('Statement', shorten=True)

    # if the node is a SameAsCluster
    def is_same_as_cluster(self):
        return self.has_type('SameAsCluster', shorten=True)

    # if the node is a ClusterMembership statement
    def is_cluster_membership(self):
        return self.has_type('ClusterMembership', shorten=True)

    # if the node has a predicate relation to a node with label pred
    def has_predicate(self, pred, shorten=False):
        return pred in self.get('predicate', shorten=shorten)

    # if the node has a subject relation to a node with label subj
    def has_subject(self, subj, shorten=False):
        return subj in self.get('subject', shorten=shorten)

    # if the node has an object relation to a node with label obj
    def has_object(self, obj, shorten=False):
        return obj in self.get("object", shorten=shorten)

    # if the node is a Statement specifying the type of node_label
    def is_type_statement(self, node_label=None):
        if self.is_statement() and self.has_predicate('type', shorten=True):
            if node_label is None or self.has_subject(node_label, shorten=False):
                return True
        return False


################################
# info classes: returned by AidaGraph,
# include node info as well as pre-parsed domain-specific info

# a typing statement
class AidaTypeInfo:
    def __init__(self, type_node):
        # type labels of a typing statement
        self.type_node = type_node
        self.type_labels = self.type_node.get('object', shorten=True)


# a node's neighbor, with the edge label between them and the direction of the edge
class AidaNeighborInfo:
    def __init__(self, node_label, neighbor_node_label, role, direction):
        self.node_label = node_label
        self.neighbor_node_label = neighbor_node_label
        self.role = role
        assert direction in ['<', '>']
        self.direction = direction

    def inverse_direction(self):
        if self.direction == '<':
            return '>'
        else:
            return '<'

    def __str__(self):
        return '{}{}{} {}'.format(
            self.direction, RDFNode.short_label(self.role), self.direction,
            RDFNode.short_label(self.neighbor_node_label))


# characterization of an entity or event in terms of its types, arguments, and events
class AidaWhoisInfo:
    def __init__(self, node: AidaNode):
        self.node = node
        # a mapping from types to corresponding confidence levels
        self.type_info = {}
        # a list of (pred, subj, AidaWhoisInfo) tuples
        self.in_edge_info = []
        # a list of (pred, obj, AidaWhoisInfo) tuples
        self.out_edge_info = []

    # for each type of this ere, keep only the maximum observed confidence level,
    # but do be prepared to keep multiple types
    def add_type(self, type_info: AidaTypeInfo, conf_levels):
        for type_label in type_info.type_labels:
            self.type_info[type_label] = max(
                max(conf_levels), self.type_info.get(type_label, 0))

    def add_in_edge(self, pred, subj, subj_whois_info):
        self.in_edge_info.append((pred, subj, subj_whois_info))

    def add_out_edge(self, pred, obj, obj_whois_info):
        self.out_edge_info.append((pred, obj, obj_whois_info))

    def prettyprint(self, indent=0, omit=None):
        if omit is None:
            omit = []

        whois_str = ''

        # node type and predicate
        if self.node is not None:
            whois_str += '{}Node: {}'.format('\t' * indent, self.node.short_name())

            if self.node.is_statement():
                whois_str += '\n{}pred: {}'.format(
                    '\t' * indent, ', '.join(self.node.get('predicate', shorten=True)))
            else:
                whois_str += '\n{}isa: {}'.format(
                    '\t' * indent, ', '.join(self.node.get('type', shorten=True)))
            if self.node.description is not None:
                whois_str += '\n{}descr: {}'.format('\t' * indent, self.node.description)

        # type info
        if len(self.type_info) > 0:
            whois_str += '\n{}types: {}'.format(
                '\t' * indent,
                ', '.join(t + '(conf=' + str(c) + ')' for t, c in self.type_info.items()))

        # incoming edges
        if len(self.in_edge_info) > 0:
            for pred, node, whois_info in self.in_edge_info:
                if node.name not in omit:
                    whois_str += '\n{}<{}< {}'.format('\t' * indent, pred, node.shortname())
                    whois_str += '\n' + whois_info.prettyprint(
                        indent=indent + 1, omit=omit + [self.node.name])

        # outgoing edges
        if len(self.out_edge_info) > 0:
            for pred, node, whois_info in self.out_edge_info:
                if node.name not in omit:
                    whois_str += '\n{}>{}> {}'.format('\t' * indent, pred, node.shortname())
                    whois_str += '\n' + whois_info.prettyprint(
                        indent=indent + 1, omit=omit + [self.node.name])


class AidaGraph(RDFGraph):

    def __init__(self, node_cls=AidaNode):
        super().__init__(node_cls=node_cls)

    # judge whether a node label exists in the graph
    def has_node(self, node_label):
        return node_label in self.node_dict

    # access method for a single node by its label
    def get_node(self, node_label):
        return self.node_dict.get(node_label, None)

    # iterator over the nodes.
    # optionally with a restriction on the type of the nodes returned
    def nodes(self, target_type=None) -> Iterable[AidaNode]:
        for node in self.node_dict.values():
            if target_type is None or node.has_type(target_type):
                yield node

    # given a node label, and a pred, return the list of object that go with it,
    # can be viewed as a composition of AidaGraph.get_node and RDFNode.get
    def get_node_objs(self, node_label, pred, shorten=False):
        node = self.get_node(node_label)
        if node:
            return node.get(pred=pred, shorten=shorten)
        else:
            return []

    # confidence level associated with a node given by its name
    def confidence_of(self, node_label):
        if not self.has_node(node_label):
            return None
        conf_levels = set()

        for conf_label in self.get_node_objs(node_label, 'confidence'):
            for c_value in self.get_node_objs(conf_label, 'confidenceValue'):
                conf_levels.add(float(c_value))
        for just_label in self.get_node_objs(node_label, 'justifiedBy'):
            for conf_label in self.get_node_objs(just_label, 'confidence'):
                for c_value in self.get_node_objs(conf_label, 'confidenceValue'):
                    conf_levels.add(float(c_value))

        return conf_levels

    # iterator over types for a particular entity/event/relation
    # yields AidaTypeInfo objects that give access ot the whole typing node
    def types_of(self, node_label):
        if self.has_node(node_label):
            for pred, subjs in self.get_node(node_label).in_edge.items():
                for subj in subjs:
                    subj_node = self.get_node(subj)
                    if subj_node and subj_node.is_type_statement(node_label):
                        yield AidaTypeInfo(subj_node)

    # iterator over KB links and associated confidence levels of an ERE or SameAsCluster node
    def kb_links_of(self, node_label):
        if self.has_node(node_label):
            for link_label in self.get_node_objs(node_label, 'link'):
                link_node = self.get_node(link_label)

                link_target = next(iter(link_node.get('linkTarget', shorten=False)), None)

                conf_levels = set()
                for conf_label in link_node.get('confidence'):
                    for c_value in self.get_node_objs(conf_label, 'confidenceValue'):
                        conf_levels.add(float(c_value))
                link_conf = next(iter(conf_levels), None)

                if link_target and link_conf:
                    yield str(link_target), link_conf

    # iterator over the labels of adjacent statements of an ERE node
    def adjacent_stmts_of_ere(self, node_label):
        node = self.get_node(node_label)
        if not node or not node.is_ere():
            return

        # check all the neighbor nodes for whether they are statements
        for pred, subjs in node.in_edge.items():
            for subj in subjs:
                if self.has_node(subj) and self.get_node(subj).is_statement():
                    yield subj

    # iterator over names of an ERE node
    def names_of_ere(self, node_label):
        node = self.get_node(node_label)
        if not node or not node.is_ere():
            return

        for name in node.get('hasName'):
            yield name.strip()
        # in M09 evaluation, some names are stored in the skos:prefLabel field of justifications
        for just_label in node.get('justifiedBy'):
            for name in self.get_node_objs(just_label, 'prefLabel'):
                yield name.strip()

    # iterator over mentions associated with the statement node
    def mentions_associated_with(self, node_label):
        node = self.get_node(node_label)
        if not node or not node.is_statement():
            return

        for just_label in node.get('justifiedBy'):
            for private_label in self.get_node_objs(just_label, 'privateData'):
                for json_str in self.get_node_objs(private_label, 'jsonContent'):
                    json_obj = json.loads(json_str)
                    if 'mention' in json_obj:
                        yield json_obj['mention']

    # iterator over provenances associate with the statement node
    def provenances_associated_with(self, node_label):
        node = self.get_node(node_label)
        if not node or not node.is_statement():
            return

        for private_label in node.get('privateData'):
            for json_str in self.get_node_objs(private_label, 'jsonContent'):
                json_obj = json.loads(json_str)
                if 'provenance' in json_obj:
                    for provenance in json_obj['provenance']:
                        yield provenance

    # iterator over source document ids associate with the statement node
    def sources_associated_with(self, node_label):
        node = self.get_node(node_label)
        if not node or not node.is_statement():
            return

        for just_label in node.get('justifiedBy'):
            for source in self.get_node_objs(just_label, 'source'):
                yield source

    # iterator over justifications associated with a node
    def justifications_associated_with(self, node_label):
        if self.has_node(node_label):
            for just_label in self.get_node_objs(node_label, 'justifiedBy'):
                just_node = self.get_node(just_label)
                if just_node.has_type('CompoundJustification', shorten=True):
                    for just_label_2 in just_node.get('containedJustification'):
                        just = self._non_compound_justification(just_label_2)
                        if just is not None:
                            yield just
                else:
                    # not a compound justification
                    just = self._non_compound_justification(just_label)
                    if just is not None:
                        yield just

    def _non_compound_justification(self, just_label):
        just_node = self.get_node(just_label)
        if not just_node:
            return None

        # type
        if just_node.has_type('TextJustification'):
            just_type = 'TextJustification'
        elif just_node.has_type('ImageJustification'):
            just_type = 'ImageJustification'
        elif just_node.has_type('KeyFrameVideoJustification'):
            just_type = 'KeyFrameVideoJustification'
        elif just_node.has_type('VideoJustification'):
            just_type = 'VideoJustification'
        else:
            return None

        # sources
        source = next(iter(just_node.get('source')), None)
        source_document = next(iter(just_node.get('sourceDocument')), None)

        just_dict = {'type': just_type, 'source': source, 'sourceDocument': source_document}

        # location
        if just_type == 'TextJustification':
            start_offset = next(iter(just_node.get('startOffset')), None)
            end_offset = next(iter(just_node.get('endOffsetInclusive')), None)
            just_dict['startOffset'] = int(start_offset) if start_offset else None
            just_dict['endOffsetInclusive'] = int(end_offset) if end_offset else None

        elif just_type == 'ImageJustification':
            just_dict['boundingBox'] = next(iter(self._bounding_boxes(just_node)), None)

        elif just_type == 'KeyFrameVideoJustification':
            just_dict['keyFrame'] = next(iter(just_node.get('keyFrame')), None)
            just_dict['boundingBox'] = next(iter(self._bounding_boxes(just_node)), None)

        # TODO: add VideoJustification for events and relations
        elif just_type == 'VideoJustification':
            raise NotImplementedError

        return just_dict

    def _bounding_boxes(self, just_node: AidaNode):
        for bb_label in just_node.get('boundingBox'):
            lower_right_x = next(iter(self.get_node_objs(bb_label, 'boundingBoxLowerRightX')), None)
            lower_right_y = next(iter(self.get_node_objs(bb_label, 'boundingBoxLowerRightY')), None)
            upper_left_x = next(iter(self.get_node_objs(bb_label, 'boundingBoxUpperLeftX')), None)
            upper_left_y = next(iter(self.get_node_objs(bb_label, 'boundingBoxUpperLeftY')), None)
            yield {
                'LowerRightX': int(lower_right_x) if lower_right_x else None,
                'LowerRightY': int(lower_right_y) if lower_right_y else None,
                'UpperLeftX': int(upper_left_x) if upper_left_x else None,
                'UpperLeftY': int(upper_left_y) if upper_left_y else None
            }

    def times_associated_with(self, node_label):
        node = self.get_node(node_label)
        if not node or not (node.is_event() or node.is_relation()):
            return

        for time_label in node.get('ldcTime'):
            start_times = [
                self._time_struct(start) for start in self.get_node_objs(time_label, 'start')]
            end_times = [self._time_struct(end) for end in self.get_node_objs(time_label, 'end')]
            yield {
                'start': [t for t in start_times if t is not None],
                'end': [t for t in end_times if t is not None]
            }

    # given an LDCTimeComponent, parse out its pieces
    def _time_struct(self, node_label):
        # each field should have at most one value, otherwise it's violating the restricted AIF
        time_type = next(iter(self.get_node_objs(node_label, 'timeType', shorten=True)), None)
        # if time_type is not AFTER or BEFORE, this is a invalid ldcTime struct, return None
        if time_type not in ['AFTER', 'BEFORE']:
            return None

        time_struct = {'timeType': time_type}

        for key in ['year', 'month', 'day', 'hour', 'minute']:
            values = [self._parse_xsd_date(key, val) for val in self.get_node_objs(node_label, key)]
            values = list(set(s for s in values if s is not None))
            if len(values) > 0:
                time_struct[key] = values[0]

        return time_struct

    # parsing out pieces of dates by hand, as I cannot find any tool that will do this
    @staticmethod
    def _parse_xsd_date(key, val):
        if key == 'year':
            extract = str(val).split('-')[0]
            if extract.isdigit():
                return int(extract)
            else:
                return None
        elif key == 'month':
            if not val.startswith('--') or not val[2:].isdigit():
                return None
            return int(val[2:])
        elif key == 'day':
            if not val.startswith('---') or not val[3:].isdigit():
                return None
            return int(val[3:])
        else:
            if not val.isdigit():
                return None
            return int(val)

    # iterator over source document ids associate with a typing statement,
    # this handles both source document information from the statement node,
    # as well as from the subject ERE node (for RPI data)
    def sources_associated_with_typing_stmt(self, node_label):
        node = self.get_node(node_label)
        if not node or not node.is_type_statement():
            return

        # iterator over source document information from the statement node
        for just_label in node.get('justifiedBy'):
            for source in self.get_node_objs(just_label, 'source'):
                yield source

        # iterator over source document information from the subject ERE node
        subj_label = next(iter(node.get('subject')))
        if self.has_node(subj_label):
            for just_label in self.get_node_objs(subj_label, 'justifiedBy'):
                for source in self.get_node_objs(just_label, 'source'):
                    yield source

    # iterator over hypotheses supported / partially supported / contradicted by the statement node
    def hypotheses_associated_with(self, node_label, hyp_rel):
        hyp_rel_labels = [
            'hypothesis',  # supported
            'partial',  # partially supported
            'contradicts'  # contradicted
        ]

        assert hyp_rel in hyp_rel_labels

        node = self.get_node(node_label)
        if not node or not node.is_statement():
            return

        for private_label in node.get('privateData'):
            for json_str in self.get_node_objs(private_label, 'jsonContent'):
                json_obj = json.loads(json_str)
                if hyp_rel in json_obj:
                    for h in json_obj[hyp_rel]:
                        yield h

    def hypotheses_supported(self, node_label):
        return self.hypotheses_associated_with(node_label, 'hypothesis')

    def hypotheses_partially_supported(self, node_label):
        return self.hypotheses_associated_with(node_label, 'partial')

    def hypotheses_contradicted(self, node_label):
        return self.hypotheses_associated_with(node_label, 'contradicts')

    # iterator over conflicting hypotheses between two statements
    def conflicting_hypotheses(self, node_label_1, node_label_2):
        if not self.has_node(node_label_1) or \
                not self.get_node(node_label_1).is_statement():
            return

        if not self.has_node(node_label_2) or \
                not self.get_node(node_label_2).is_statement():
            return

        # hypotheses fully / partially supported by node_label_1
        supporting_hyp_1 = set(self.hypotheses_associated_with(node_label_1, 'hypothesis'))
        supporting_hyp_1.update(set(self.hypotheses_associated_with(node_label_1, 'partial')))
        # hypotheses contradicted by node_label_1
        contradicting_hyp_1 = set(self.hypotheses_associated_with(node_label_1, 'contradicts'))

        # hypotheses fully / partially supported by node_label_2
        supporting_hyp_2 = set(self.hypotheses_associated_with(node_label_2, 'hypothesis'))
        supporting_hyp_2.update(set(self.hypotheses_associated_with(node_label_2, 'partial')))
        # hypotheses contradicted by node_label_2
        contradicting_hyp_2 = set(self.hypotheses_associated_with(node_label_2, 'contradicts'))

        for h in supporting_hyp_1.intersection(contradicting_hyp_2):
            yield h
        for h in supporting_hyp_2.intersection(contradicting_hyp_1):
            yield h

    # iterator over neighbors of a node
    # that mention the label of the entity, or whose label is mentioned
    # in the entry of the entity
    # yields AidaNeighborInfo objects
    def neighbors_of(self, node_label):
        node = self.get_node(node_label)
        if not node:
            return

        for pred, objs in node.out_edge.items():
            for obj in objs:
                yield AidaNeighborInfo(node_label, obj, pred, '>')
        for pred, subjs in node.in_edge.items():
            for subj in subjs:
                yield AidaNeighborInfo(node_label, subj, pred, '<')

    # output a characterization of a node:
    # what is its type,
    # what events is it involved in (for an entity),
    # what arguments does it have (for an event)
    def whois(self, node_label, follow=2):
        node = self.get_node(node_label)
        if not node:
            return None

        whois_info = AidaWhoisInfo(node)

        # we do have an entry for this node.
        # determine its types
        for type_info in self.types_of(node_label):
            conf_levels = self.confidence_of(type_info.type_node.name)
            if conf_levels is not None:
                whois_info.add_type(type_info, conf_levels)

        if follow > 0:
            # we were asked to also explore this node's neighbors
            # explore incoming edges
            for pred, subjs in node.in_edge.items():
                for subj in subjs:
                    if self.has_node(subj):
                        subj_node = self.get_node(subj)
                        if subj_node.is_type_statement() or subj_node.is_kb_entry_statement():
                            # don't re-record typing nodes
                            continue
                        elif subj_node.is_statement():
                            subj_whois_info = self.whois(subj, follow=follow - 1)
                            whois_info.add_in_edge(pred, subj_node, subj_whois_info)

            # explore outgoing edges
            for pred, objs in node.out_edge.items():
                for obj in objs:
                    if self.has_node(obj):
                        obj_node = self.get_node(obj)
                        if obj_node.is_statement() or obj_node.is_ere():
                            obj_whois_info = self.whois(obj, follow=follow - 1)
                            whois_info.add_out_edge(pred, obj_node, obj_whois_info)

        return whois_info

    # traverse: explore the whole reachable graph starting from start_node_label,
    # yields pairs (node_label, path)
    # where path is a list of AidaNeighborInfo objects, starting from start_node_label
    def traverse(self, start_node_label, omit_roles=omit_labels):
        if not self.has_node(start_node_label):
            return

        to_visit = deque([(start_node_label, [])])
        edges_visited = set()

        omit_roles = omit_roles or set()

        while to_visit:
            current_label, current_path = to_visit.popleft()

            yield current_label, current_path

            for neighbor_info in self.neighbors_of(current_label):
                if neighbor_info.role in omit_roles:
                    continue
                edges = [
                    (current_label, neighbor_info.role,
                     neighbor_info.direction, neighbor_info.neighbor_node_label),
                    (neighbor_info.neighbor_node_label, neighbor_info.role,
                     neighbor_info.inverse_direction(), current_label)
                ]

                if any(edge in edges_visited for edge in edges):
                    continue
                else:
                    to_visit.append(
                        (neighbor_info.neighbor_node_label, current_path + [neighbor_info]))
                    for edge in edges:
                        edges_visited.add(edge)
