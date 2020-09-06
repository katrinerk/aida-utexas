"""
Author: Pengxiang Cheng, Aug 2020
- Adapted from the legacy soin_processing package by Eric.
- Statement of information needs.
- Each statement of information need contains a list of frames, where each frame contains a list of
edges, a list of temporal information, and a list of entry point specifications.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List
from xml.etree import ElementTree

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

    def to_list(self):
        return [self.subject, self.predicate, self.object]


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

    def to_dict(self, temporal_info_dict):
        return {
            'temporal': temporal_info_dict,
            'queryConstraints': [edge.to_list() for edge in self.edges],
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
                time_dict[elem.tag] = elem.text.strip() if elem.text else None
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


@dataclass
class SOIN:
    file_path: str
    id: str
    frames: List[Frame] = field(default_factory=list)
    temporal_info_list: List[TemporalInfo] = field(default_factory=list)
    entrypoints: List[EntryPoint] = field(default_factory=list)

    def frames_to_json(self):
        temporal_info_dict = {}
        for temporal_info in self.temporal_info_list:
            temporal_info_dict[temporal_info.subject] = {
                'start_time': asdict(temporal_info.start_time),
                'end_time': asdict(temporal_info.end_time)
            }
        return [frame.to_dict(temporal_info_dict) for frame in self.frames]

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
