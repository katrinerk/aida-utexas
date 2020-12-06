"""
Author: Pengxiang Cheng, Aug 2020
- Adapted from the legacy soin_processing package by Eric.
- Statement of information needs.
- Each statement of information need contains a list of frames, where each frame contains a list of
edges, a list of temporal information, and a list of entry point specifications.

Update: Pengxiang Cheng, Sep 2020
- Merge the resolve_all_entrypoints method from process_soin
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List
from xml.etree import ElementTree

from aida_utexas.aif import AidaGraph
from aida_utexas.soin.entry_point import EntryPoint


@dataclass
class Edge:
    id: str
    subject: str
    predicate: str
    object: str
    objectType: str = None

    @classmethod
    def from_xml(cls, xml_elem):
        edge_dict = {'id': xml_elem.attrib['id']}

        for elem in xml_elem:
            if elem.tag in ['subject', 'predicate', 'object', 'objectType']:
                edge_dict[elem.tag] = elem.text.strip() if elem.text else None
            else:
                logging.warning(f'Unexpected tag {elem.tag} in SOIN')

        return cls(**edge_dict)

    def to_json(self):
        return [self.id, self.subject, self.predicate, self.object, self.objectType]


@dataclass
class Frame:
    id: str
    edges: List[Edge] = field(default_factory=list)

    @classmethod
    def from_xml(cls, xml_elem):
        frame_dict = {'id': xml_elem.attrib['id'], 'edges': []}

        edges = xml_elem[0]
        for elem in edges:
            frame_dict['edges'].append(Edge.from_xml(elem))

        return cls(**frame_dict)

    def to_json(self):
        return {
            'frame_id': self.id,
            'edges': [edge.to_json() for edge in self.edges],
        }


@dataclass
class Time:
    year: int = None
    month: int = None
    day: int = None
    hour: int = None
    minute: int = None

    @classmethod
    def from_xml(cls, xml_elem):
        time_dict = {}

        for elem in xml_elem:
            if elem.tag in ['year', 'month', 'day', 'hour', 'minute']:
                if elem.text is None:
                    time_dict[elem.tag] = None
                # NOBUG: if the field is a variable name starting with ?,
                #  that indicates the analyst is requesting temporal information for the event or
                #  relation. We return a negative number to indicate that.
                elif elem.text.strip().startswith('?'):
                    time_dict[elem.tag] = -999
                else:
                    time_dict[elem.tag] = int(elem.text.strip())
            else:
                logging.warning(f'Unexpected tag {elem.tag} in SOIN')

        return cls(**time_dict)


@dataclass
class TemporalInfo:
    subject: str
    start_time: Time
    end_time: Time

    @classmethod
    def from_xml(cls, xml_elem):
        temporal_info_dict = {}

        for elem in xml_elem:
            if elem.tag == 'subject':
                temporal_info_dict['subject'] = elem.text.strip() if elem.text else None
            elif elem.tag in ['start_time', 'end_time']:
                temporal_info_dict[elem.tag] = Time.from_xml(elem)
            else:
                logging.warning(f'Unexpected tag {elem.tag} in SOIN')

        return cls(**temporal_info_dict)

    def to_json(self):
        return {
            self.subject: {
                'start_time': asdict(self.start_time),
                'end_time': asdict(self.end_time)
            }
        }


@dataclass
class SOIN:
    # the path of the SoIN xml file
    file_path: str
    # the SoIN id
    id: str
    # a list of frames, each with a frame id and a list of edges
    frames: List[Frame] = field(default_factory=list)
    # a list of temporal constraints
    temporal_info_list: List[TemporalInfo] = field(default_factory=list)
    # a list of entry point variables, each characterized by one or more typed descriptors
    entrypoints: List[EntryPoint] = field(default_factory=list)
    # a dictionary of ERE matches and weights for each entry point variable
    ep_matches_dict: Dict = None

    def to_json(self):
        temporal_dict = {}
        for temporal_info in self.temporal_info_list:
            temporal_dict.update(temporal_info.to_json())

        return {
            'soin_id': self.id,
            'ep_matches_dict': self.ep_matches_dict,
            'frames': [frame.to_json() for frame in self.frames],
            'temporal': temporal_dict
        }

    @classmethod
    def parse(cls, file_path: str, dup_kbid_mapping: Dict = None):
        """
        This function will process the XML input and retrieve:
            (1) The specified frames
            (2) The temporal information list
            (3) The entrypoint specifications
        """

        tree = ElementTree.parse(file_path)
        root = tree.getroot()

        # Construct some data structures to hold key information.
        soin_dict = {
            'file_path': str(file_path),
            'id': root.attrib['id'],
            'frames': [],
            'temporal_info_list': [],
            'entrypoints': [],
        }

        # Traverse the XML tree
        for div in root:  # information_need
            # frame definitions
            if div.tag == 'frames':  # information_need|frames
                for elem in div:  # information_need|frames|frame
                    soin_dict['frames'].append(Frame.from_xml(elem))

            # temporal information definitions
            elif div.tag == 'temporal_info_list':  # information_need|temporal_info_list
                for elem in div:  # information_need|temporal_info_list|temporal_info
                    soin_dict['temporal_info_list'].append(TemporalInfo.from_xml(elem))

            # entry point specifications
            elif div.tag == "entrypoints":  # information_need|entrypoints
                for elem in div:  # information_need|entrypoints|entrypoint
                    soin_dict['entrypoints'].append(EntryPoint.from_xml(elem, dup_kbid_mapping))

            else:
                logging.warning(f'Unexpected tag {div.tag} in SOIN')

        return SOIN(**soin_dict)

    def resolve(self, aida_graph: AidaGraph, ere_to_prototypes: Dict, max_matches: int):
        self.ep_matches_dict = {}

        for entrypoint in self.entrypoints:
            self.ep_matches_dict[entrypoint.node] = entrypoint.resolve(
                aida_graph, ere_to_prototypes, max_matches)
