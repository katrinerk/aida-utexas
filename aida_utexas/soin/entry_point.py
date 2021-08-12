"""
Author: Pengxiang Cheng, Aug 2020
- Adapted from the legacy soin_processing package by Eric.
- Entry point specifications in statement of information needs.
- Each entry point has a list of TypedDescriptors, where each TypedDescriptor consists of an EntType
specification and a (String|KB|Text|Image|Video)Descriptor.

Update: Pengxiang Cheng, Sep 2020
- Merge the find_entrypoint method from process_soin
"""

import itertools
import logging
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Dict, List, Tuple
from xml.etree.ElementTree import Element
import edit_distance

from aida_utexas.aif import AidaGraph

match_score_weight = {'type': 1, 'descriptor': 10}


@dataclass
class EntType:
    target_type: List[str] = field(default_factory=list)

    @classmethod
    def from_xml(cls, xml_elem: Element):
        target_type = xml_elem.text.strip().split('.') if xml_elem.text else []
        while len(target_type) < 3:
            target_type.append('')

        return cls(target_type)

    def match_score(self, ere_type: str = None):
        if not ere_type:
            return 0

        ere_type = ere_type.strip().split('.')

        num_levels = 0
        num_matches = 0

        for target_t, ere_t in itertools.zip_longest(self.target_type, ere_type):
            if target_t:
                num_levels += 1
                if ere_t and target_t.lower() == ere_t.lower():
                    num_matches += 1

        if num_levels == 0:
            return 0
        return num_matches / num_levels * 100


# From the M36 evaluation plan, it seems that only StringDescriptor and KBDescriptor will be used
@dataclass
class Descriptor:
    descriptor_type: str = None

    @classmethod
    def from_xml(cls, xml_elem: Element):
        if xml_elem.tag == 'string_descriptor':
            return StringDescriptor.from_xml(xml_elem)
        elif xml_elem.tag == 'kb_descriptor':
            return KBDescriptor.from_xml(xml_elem)
        elif xml_elem.tag == 'text_descriptor':
            return TextDescriptor.from_xml(xml_elem)
        elif xml_elem.tag == 'image_descriptor':
            return ImageDescriptor.from_xml(xml_elem)
        elif xml_elem.tag == 'video_descriptor':
            return VideoDescriptor.from_xml(xml_elem)
        else:
            logging.warning(f'Unexpected tag {xml_elem.tag} in SOIN')
            return None

    def match_score(self, node_label: str, graph: AidaGraph):
        raise NotImplementedError


@dataclass
class StringDescriptor(Descriptor):
    descriptor_type: str = 'String'
    name_string: str = None

    @classmethod
    def from_xml(cls, xml_elem: Element):
        string_descriptor_dict = {}

        for elem in xml_elem:
            if elem.tag == 'name_string':
                string_descriptor_dict['name_string'] = elem.text.strip() if elem.text else None
            else:
                logging.warning(f'Unexpected tag {elem.tag} in SOIN')

        return cls(**string_descriptor_dict)

    def edit_dist_string_similarity(self, str1, str2):
        total_length = len(str1) + len(str2)
        sm = edit_distance.SequenceMatcher(a=str1, b=str2)
        dist = sm.distance()
        return round((total_length - dist) / total_length * 100, 2)

    def match_score(self, node_label: str, graph: AidaGraph):
        string_match_rates = []
        for ere_name in graph.names_of_ere(node_label):
            edit_dist_ratio = self.edit_dist_string_similarity(ere_name, self.name_string)
            string_match_rates.append(edit_dist_ratio)
        if len(string_match_rates) == 0:
            return 0
        return max(string_match_rates)


@dataclass
class KBDescriptor(Descriptor):
    descriptor_type: str = 'KB'
    kbid: str = None
    dup_kbid: str = None

    @classmethod
    def from_xml(cls, xml_elem: Element):
        kb_descriptor_dict = {}

        for elem in xml_elem:
            if elem.tag == 'kbid':
                kb_descriptor_dict['kbid'] = elem.text.strip() if elem.text else None
            else:
                logging.warning(f'Unexpected tag {elem.tag} in SOIN')

        return cls(**kb_descriptor_dict)

    def update_dup_kbid(self, dup_kbid_mapping: Dict):
        if self.kbid in dup_kbid_mapping:
            self.dup_kbid = dup_kbid_mapping[self.kbid]

    def match_score(self, node_label: str, graph: AidaGraph):
        for link_target, link_conf in graph.kb_links_of(node_label):
            if link_target == self.kbid or link_target == self.dup_kbid:
                return 100
        return 0


@dataclass
class TextDescriptor(Descriptor):
    descriptor_type: str = 'Text'
    doceid: str = None
    start: int = None
    end: int = None

    @classmethod
    def from_xml(cls, xml_elem: Element):
        text_descriptor_dict = {}

        for elem in xml_elem:
            if elem.tag == 'doceid':
                text_descriptor_dict['doceid'] = elem.text.strip() if elem.text else None
            elif elem.tag in ['start', 'end']:
                text_descriptor_dict[elem.tag] = int(elem.text.strip()) if elem.text else None
            else:
                logging.warning(f'Unexpected tag {elem.tag} in SOIN')

        return cls(**text_descriptor_dict)

    def span_overlap(self, just_start: int, just_end: int):
        if just_end <= self.start or self.end <= just_start:
            return 0

        overlap_start = max(self.start, just_start)
        overlap_end = min(self.end, just_end)

        overlap_ratio_1 = (overlap_end - overlap_start) / (self.end - self.start)
        overlap_ratio_2 = (overlap_end - overlap_start) / (just_end - just_start)

        return (overlap_ratio_1 + overlap_ratio_2) / 2 * 100

    def match_score(self, node_label: str, graph: AidaGraph):
        logging.warning('TextDescriptor is not guaranteed to work correctly, as there is '
                        'no TextDescriptor in sample SINs, use at your own risk')

        all_scores = []

        for just in graph.justifications_associated_with(node_label):
            just_score = 0
            if just['type'] == 'TextJustification':
                if self.doceid == just['source']:
                    just_score += 10
                    just_score += 0.9 * self.span_overlap(
                        just['startOffset'], just['endOffsetInclusive'])
            all_scores.append(just_score)

        raise max(all_scores)


@dataclass
class ImageDescriptor(Descriptor):
    descriptor_type: str = 'Image'
    doceid: str = None
    topleft: Tuple[int, int] = None
    bottomright: Tuple[int, int] = None

    @classmethod
    def from_xml(cls, xml_elem: Element):
        image_descriptor_dict = {}

        for elem in xml_elem:
            if elem.tag == 'doceid':
                image_descriptor_dict['doceid'] = elem.text.strip() if elem.text else None
            elif elem.tag in ['topleft', 'bottomright']:
                image_descriptor_dict[elem.tag] = (int(x) for x in elem.text.strip().split(',')) \
                    if elem.text else None
            else:
                logging.warning(f'Unexpected tag {elem.tag} in SOIN')

        return cls(**image_descriptor_dict)

    def bounding_box_overlap(self, just_bounding_box: Dict[str, int]):
        just_topleft = (just_bounding_box['UpperLeftX'], just_bounding_box['UpperLeftY'])
        just_bottomright = (just_bounding_box['LowerRightX'], just_bounding_box['LowerRightY'])

        if just_bottomright[0] < self.topleft[0] or self.bottomright[0] < just_topleft[0]:
            return 0
        if just_topleft[1] < self.bottomright[1] or self.topleft[1] < just_bottomright[1]:
            return 0

        overlap_topleft_x = max(self.topleft[0], just_topleft[0])
        overlap_topleft_y = min(self.topleft[1], just_topleft[1])
        overlap_bottomright_x = min(self.bottomright[0], just_bottomright[0])
        overlap_bottomright_y = max(self.bottomright[1], just_bottomright[1])

        target_area = (self.bottomright[0] - self.topleft[0]) * (
                self.topleft[1] - self.bottomright[0])
        just_area = (just_bottomright[0] - just_topleft[0]) * (
                just_topleft[1] - just_bottomright[0])
        overlap_area = (overlap_bottomright_x - overlap_topleft_x) * (
                overlap_topleft_y - overlap_bottomright_y)

        overlap_ratio_1 = overlap_area / target_area
        overlap_ratio_2 = overlap_area / just_area

        return (overlap_ratio_1 + overlap_ratio_2) / 2 * 100

    def match_score(self, node_label: str, graph: AidaGraph):
        logging.warning('ImageDescriptor is not guaranteed to work correctly, as there is '
                        'no ImageDescriptor in sample SINs, use at your own risk')

        all_scores = []

        for just in graph.justifications_associated_with(node_label):
            just_score = 0
            if just['type'] == 'ImageJustification':
                if self.doceid == just['source']:
                    just_score += 10
                    just_score += 0.9 * self.bounding_box_overlap(just['boundingBox'])

            all_scores.append(just_score)

        raise max(all_scores)


@dataclass
class VideoDescriptor(ImageDescriptor):
    descriptor_type: str = 'Video'
    doceid: str = None
    keyframeid: str = None
    topleft: Tuple[int, int] = None
    bottomright: Tuple[int, int] = None

    @classmethod
    def from_xml(cls, xml_elem: Element):
        video_descriptor_dict = {}

        for elem in xml_elem:
            if elem.tag in ['doceid', 'keyframeid']:
                video_descriptor_dict[elem.tag] = elem.text.strip() if elem.text else None
            elif elem.tag in ['topleft', 'bottomright']:
                video_descriptor_dict[elem.tag] = (int(x) for x in elem.text.strip().split(',')) \
                    if elem.text else None
            else:
                logging.warning(f'Unexpected tag {elem.tag} in SOIN')

        return cls(**video_descriptor_dict)

    def match_score(self, node_label: str, graph: AidaGraph):
        logging.warning('VideoDescriptor is not guaranteed to work correctly, as there is '
                        'no VideoDescriptor in sample SINs, use at your own risk')

        all_scores = []

        for just in graph.justifications_associated_with(node_label):
            just_score = 0
            if just['type'] == 'ImageJustification':
                if self.doceid == just['source']:
                    just_score += 10
                    if self.keyframeid == just['keyFrame']:
                        just_score += 10
                        just_score += 0.8 * self.bounding_box_overlap(just['boundingBox'])

            all_scores.append(just_score)

        raise max(all_scores)


@dataclass
class TypedDescriptor:
    enttype: EntType = None
    descriptor: Descriptor = None

    @classmethod
    def from_xml(cls, xml_elem: Element, dup_kbid_mapping: Dict = None):
        typed_descriptor_dict = {}

        for elem in xml_elem:
            if elem.tag == 'enttype':
                typed_descriptor_dict['enttype'] = EntType.from_xml(elem)
            else:
                descriptor = Descriptor.from_xml(elem)
                if isinstance(descriptor, KBDescriptor) and dup_kbid_mapping:
                    descriptor.update_dup_kbid(dup_kbid_mapping)

                if descriptor:
                    if 'descriptor' in typed_descriptor_dict:
                        logging.warning(f'Found more than one descriptors in a typed descriptor, '
                                        f'only keeping the last one')

                    typed_descriptor_dict['descriptor'] = descriptor

        return cls(**typed_descriptor_dict)

    def descriptor_type(self):
        return self.descriptor.descriptor_type


@dataclass
class EntryPoint:
    node: str
    typed_descriptors: List[TypedDescriptor] = field(default_factory=list)

    @classmethod
    def from_xml(cls, xml_elem: Element, dup_kbid_mapping: Dict = None):
        entrypoint_dict = {'node': None, 'typed_descriptors': []}

        for elem in xml_elem:
            if elem.tag == 'node':
                entrypoint_dict['node'] = elem.text.strip() if elem.text else None
            elif elem.tag == 'typed_descriptors':
                for typed_descriptor_elem in elem:
                    entrypoint_dict['typed_descriptors'].append(
                        TypedDescriptor.from_xml(typed_descriptor_elem, dup_kbid_mapping))

        return cls(**entrypoint_dict)

    def resolve(self, aida_graph: AidaGraph, ere_to_prototypes: Dict, max_matches: int) -> List:
        """
        A function to resolve an entrypoint to the set of entity nodes that satisfy it.
        This function iterates through every node in the graph. If that node is a typing statement,
        it computes a type score (how many matches between the entity type/subtype/subsubtype) and
        a descriptor score (how many complete TypedDescriptor matches) across all TypedDescriptors.

        The function returns the set of nodes mapped to the highest scores.
        :param aida_graph: AidaGraph
        :param ere_to_prototypes: dict
        :param max_matches: int
        """
        results = {}

        for node in aida_graph.nodes():
            if node.is_type_statement():
                subj_label = next(iter(node.get('subject', shorten=False)))
                obj_label = next(iter(node.get('object', shorten=True)))

                all_scores = []

                for typed_descriptor in self.typed_descriptors:
                    type_score = 0
                    descriptor_score = 0

                    has_type = 0
                    has_descriptor = 0

                    if typed_descriptor.enttype:
                        has_type = 1
                        type_score = typed_descriptor.enttype.match_score(obj_label)

                    if typed_descriptor.descriptor:
                        has_descriptor = 1
                        descriptor_score = typed_descriptor.descriptor.match_score(
                            subj_label, aida_graph)

                        if typed_descriptor.descriptor_type() in ['Text', 'Image', 'Video']:
                            descriptor_score = max(
                                descriptor_score,
                                typed_descriptor.descriptor.match_score(node.name, aida_graph))

                    total_score = denominator = 0
                    if has_type:
                        total_score += type_score * match_score_weight['type']
                        denominator += match_score_weight['type']
                    if has_descriptor:
                        total_score += descriptor_score * match_score_weight['descriptor']
                        denominator += match_score_weight['descriptor']

                    all_scores.append(total_score / denominator)

                avg_score = sum(all_scores) / len(all_scores)

                for prototype in ere_to_prototypes[subj_label]:
                    if prototype in results:
                        results[prototype] = max(results[prototype], avg_score)
                    else:
                        results[prototype] = avg_score

        return sorted(results.items(), key=itemgetter(1), reverse=True)[:max_matches]

    def describe(self):
        """ This function returns a dictionary with entries KB and String,
        each a list of strings, characterizing all descriptors of this entry point.
        """

        strings = [ ]
        kbs = [ ]

        for typed_descriptor in self.typed_descriptors:
            if typed_descriptor.descriptor:
                if typed_descriptor.descriptor_type() == "String":
                    strings.append(typed_descriptor.descriptor.name_string)
                elif typed_descriptor.descriptor_type() == "KB":
                    kbs.append(typed_descriptor.descriptor.kbid)

        return { "String": strings, "KB": kbs }
        

        
    
