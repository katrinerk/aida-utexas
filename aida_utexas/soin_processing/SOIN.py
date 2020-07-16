import logging
import xml.etree.ElementTree as ET
from copy import deepcopy

from aida_utexas.soin_processing.TypedDescriptor import *
from aida_utexas.soin_processing.templates_and_constants import *

_LOG = logging.getLogger('read_soin')
in_path = "/Users/eholgate/Desktop/SOIN/new_SOINs/R103.xml"


def del_xx(in_time):
    if (not in_time) or (in_time == "XX"):
        return ""
    else:
        return in_time.strip()


class UnexpectedXMLTag(Exception):
    """
    This Exception will be raised if the processing script encounters unexpected information in the SOIN input.
    """
    pass


class Time:
    def __init__(self, year, month, day, hour, minute):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'year': self.year,
            'month': self.month,
            'day': self.day,
            'hour': self.hour,
            'minute': self.minute,
        }
        return str(rep)

    def time_to_dict(self):
        rep = {
            'year': self.year,
            'month': self.month,
            'day': self.day,
            'hour': self.hour,
            'minute': self.minute,
        }
        return rep


class TemporalInfo:
    def __init__(self, subject, start_time, end_time):
        self.subject = subject
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'subject': self.subject,
            'start_time': self.start_time,
            'end_time': self.end_time,
        }
        return str(rep)


class Frame:
    def __init__(self, frame_id, edge_list):
        self.id = frame_id
        self.edge_list = edge_list

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'id': self.id,
            'edge_list': self.edge_list,
        }

        return str(rep)

    def frame_to_dict(self, temporal_dict):
        query_constraints = []
        for edge in self.edge_list:
            query_constraints.append(edge.edge_to_dict())

        rep = {
            'temporal': temporal_dict,
            'queryConstraints': query_constraints,
        }

        return rep


class Edge:
    def __init__(self, edge_id, subject, predicate, obj, objType):
        self.id = edge_id
        self.subject = subject
        self.predicate = predicate
        self.obj = obj
        self.objType = objType

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'id': self.id,
            'subject': self.subject,
            'predicate': self.predicate,
            'obj': self.obj,
            'objType': self.objType,
        }
        return str(rep)

    def edge_to_dict(self):
        subj = self.subject
        obj = self.obj

        rep = [
            subj,
            self.predicate,
            obj
        ]

        return rep


class Entrypoint:
    def __init__(self, variable, typed_descriptor_list):
        self.variable = variable,
        self.typed_descriptor_list = typed_descriptor_list,

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'variable': self.variable,
            'typed_descriptor_list': self.typed_descriptor_list,
        }
        return str(rep)


class SOIN:
    def __init__(self, in_path, soin_id, frames, temporal_info, entrypoints):
        self.in_path = in_path
        self.id = soin_id
        self.frames = frames
        self.temporal_info = temporal_info
        self.entrypoints = entrypoints
        self.json = {}

    def __repr__(self):
        returnme = f"SOIN {self.soin_id}: " + self.in_path
        returnme += "Frames:\n"
        for frame in self.frames:
            returnme += "Frame: " + str(frame)
            for edge in self.frames[frame]:
                returnme += "\tEdge: " + str(edge)
                for rel in self.frames[frame][edge]:
                    returnme += "\t\t" + str(rel)
        returnme += "========================================\n\n"
        returnme += "Temporal Information:\n"
        for subject in self.temporal_info:
            returnme += "Subject: " + str(subject)
            for time in self.temporal_info[subject]:
                returnme += "\t" + str(time) + ": "
                for unit in self.temporal_info[subject][time]:
                    returnme += "\t\t" + str(unit) + ": " + self.temporal_info[subject][time][unit]
        returnme += "========================================\n\n"
        returnme += "EPs:\n"
        for ep_num, ep_dict in enumerate(self.entrypoints):
            returnme += "EP " + str(ep_num + 1) + " : "
            for k in ep_dict:
                returnme += "\t" + k + ": " + str(ep_dict[k])
        return returnme

    def temporal_info_to_dict(self):
        temporal_dict = {}
        for info in self.temporal_info:
            temporal_dict[info.subject] = {
                'start_time': info.start_time.time_to_dict(),
                'end_time': info.end_time.time_to_dict(),
            }
        return temporal_dict


def process_xml(in_path, dup_kbid_mapping=None):
    """
    This function will process the XML input and retrieve:
        (1) The specified frames
        (2) Any temporal information
        (3) The entrypoint specifications

    :return:
        :frames: a dictionary mapping frame IDs to lists of query constraint triples
        :temporal_info: a dictionary mapping start_time and end_time to dictionaries mapping temporal units to
                        constraint specifications
        :eps: a list of entrypoint representations
    """
    tree = ET.parse(in_path)
    root = tree.getroot()

    # Construct some data structures to hold key information.
    soin_dict = {
        'in_path': in_path,
        'soin_id': root.attrib['id'],
        'frames': [],
        'entrypoints': [],
        'temporal_info': [],
    }

    # Traverse the XML tree
    for div in root:  # information_need
        # Handle frame definitions
        if div.tag == 'frames':  # information_need|frames
            for frame in div:  # information_need|frames|frame
                frame_dict = {
                    'frame_id': frame.attrib['id'],
                    'edge_list': [],
                }

                for edges in frame:  # information_need|frames|frame|edges
                    for edge in edges:  # information_need|frames|frame|edges|edge
                        edge_dict = {
                            'edge_id': edge.attrib['id'],
                            'subject': None,
                            'predicate': None,
                            'obj': None,
                            'objType': None,
                        }

                        for relation in edge:  # information_need|frames|frame|edges|edge|
                            # (subject, predicate, object, objectType)
                            if relation.tag == 'subject':
                                edge_dict['subject'] = relation.text.strip()
                            elif relation.tag == "predicate":
                                edge_dict['predicate'] = relation.text.strip()
                            elif relation.tag == "object":
                                edge_dict['obj'] = relation.text.strip()
                            elif relation.tag == "objectType":
                                edge_dict['objType'] = relation.text.strip()
                            else:
                                _LOG.log(50, 'Process SOIN: Unexpected tag {} encountered'.format(relation.tag))
                                raise UnexpectedXMLTag

                        frame_dict['edge_list'].append(Edge(**edge_dict))

                    # Hand the frame representation up to be returned later
                    soin_dict['frames'].append(Frame(**frame_dict))

        # Handle temporal information definitions
        elif div.tag == 'temporal_info_list':  # information_need|temporal_info_list
            for temp_info in div:  # information_need|temporal_info_list|temporal_info
                temporal_info_dict = {
                    'subject': None,
                    'start_time': None,
                    'end_time': None,
                }

                for time in temp_info:  # information_need|temporal_info_list|temporal_info|
                    # (subject, start_time, end_time)
                    if time.tag == 'subject':
                        temporal_info_dict['subject'] = time.text.strip()

                    elif time.tag == 'start_time':
                        start_time_dict = {
                            'year': None,
                            'month': None,
                            'day': None,
                            'hour': None,
                            'minute': None,
                        }

                        for info in time:
                            if info.tag == 'year':
                                start_time_dict['year'] = del_xx(info.text)
                            elif info.tag == 'month':
                                start_time_dict['month'] = del_xx(info.text)
                            elif info.tag == 'day':
                                start_time_dict['day'] = del_xx(info.text)
                            elif info.tag == 'hour':
                                start_time_dict['hour'] = del_xx(info.text)
                            elif info.tag == 'minute':
                                start_time_dict['minute'] = del_xx(info.text)
                            else:
                                _LOG.log(50, 'Process SOIN: Unexpected tag {} encountered'.format(info.tag))
                                raise UnexpectedXMLTag

                        # Hand this up
                        temporal_info_dict['start_time'] = Time(**start_time_dict)

                    elif time.tag == 'end_time':
                        end_time_dict = {
                            'year': None,
                            'month': None,
                            'day': None,
                            'hour': None,
                            'minute': None,
                        }

                        for info in time:
                            if info.tag == 'year':
                                end_time_dict['year'] = del_xx(info.text)
                            elif info.tag == 'month':
                                end_time_dict['month'] = del_xx(info.text)
                            elif info.tag == 'day':
                                end_time_dict['day'] = del_xx(info.text)
                            elif info.tag == 'hour':
                                end_time_dict['hour'] = del_xx(info.text)
                            elif info.tag == 'minute':
                                end_time_dict['minute'] = del_xx(info.text)
                            else:
                                _LOG.log(50, 'Process SOIN: Unexpected tag {} encountered'.format(info.tag))
                                raise UnexpectedXMLTag

                        # Hand this up
                        temporal_info_dict['end_time'] = Time(**end_time_dict)

                    else:
                        _LOG.log(50, 'Process SOIN: Unexpected tag {} encountered'.format(time.tag))
                        raise UnexpectedXMLTag

                soin_dict['temporal_info'].append(TemporalInfo(**temporal_info_dict))

        elif div.tag == "entrypoints":  # information_need|entrypoints
            for ep in div:  # information_need|entrypoints|entrypoint
                ep_dict = deepcopy(EP_REP_TEMPLATE)

                for elem in ep:
                    if elem.tag == 'node':
                        ep_dict['variable'] = elem.text.strip()

                    elif elem.tag == 'typed_descriptors':
                        for typed_descriptor in elem:
                            typed_descriptor_dict = deepcopy(TYPED_DESCRIPTOR_TEMPLATE)

                            for descriptor in typed_descriptor:
                                if descriptor.tag == 'enttype':
                                    enttype = descriptor.text.strip().split('.')
                                    for i in range(3 - len(enttype)):  # Force the length of this list to be 3
                                        enttype.append('')

                                    typed_descriptor_dict['enttype'] = EntType(typ=enttype[0],
                                                                               subtype=enttype[1],
                                                                               subsubtype=enttype[2])
                                elif descriptor.tag == 'text_descriptor':
                                    text_descriptor_dict = deepcopy(TEXT_DESCRIPTOR_REP_TEMPLATE)
                                    for info in descriptor:
                                        if info.tag == "doceid":
                                            text_descriptor_dict["doceid"] = info.text.strip()

                                        elif info.tag == "start":
                                            text_descriptor_dict["start"] = info.text.strip()

                                        elif info.tag == "end":
                                            text_descriptor_dict["end"] = info.text.strip()

                                        else:
                                            _LOG.log(50,
                                                     'Process SOIN: Unexpected tag {} encountered'.format(info.tag))
                                            raise UnexpectedXMLTag
                                    typed_descriptor_dict['descriptor'] = TextDescriptor(**text_descriptor_dict)

                                elif descriptor.tag == 'string_descriptor':
                                    string_descriptor_dict = deepcopy(STRING_DESCRIPTOR_REP_TEMPLATE)
                                    for info in descriptor:
                                        if info.tag == 'name_string':
                                            string_descriptor_dict['name_string'] = info.text.strip()
                                        else:
                                            _LOG.log(50,
                                                     'Process SOIN: Unexpected tag {} encountered'.format(info.tag))
                                            raise UnexpectedXMLTag
                                    typed_descriptor_dict['descriptor'] = StringDescriptor(**string_descriptor_dict)

                                elif descriptor.tag == 'video_descriptor':
                                    video_descriptor_dict = deepcopy(VIDEO_DESCRIPTOR_REP_TEMPLATE)
                                    for info in descriptor:
                                        if info.tag == 'doceid':
                                            video_descriptor_dict['doceid'] = info.text.strip()
                                        elif info.tag == 'keyframeid':
                                            video_descriptor_dict['keyframe_id'] = info.text.strip()
                                        elif info.tag == 'topleft':
                                            video_descriptor_dict['top_left'] = info.text.strip()
                                        elif info.tag == 'bottomright':
                                            video_descriptor_dict['bottom_right'] = info.text.strip()
                                        else:
                                            _LOG.log(50,
                                                     'Process SOIN: Unexpected tag {} encountered'.format(info.tag))
                                            raise UnexpectedXMLTag
                                    typed_descriptor_dict['descriptor'] = VideoDescriptor(**video_descriptor_dict)

                                elif descriptor.tag == 'image_descriptor':
                                    image_descriptor_dict = deepcopy(IMAGE_DESCRIPTOR_REP_TEMPLATE)
                                    for info in descriptor:
                                        if info.tag == 'doceid':
                                            image_descriptor_dict['doceid'] = info.text.strip()
                                        elif info.tag == 'topleft':
                                            image_descriptor_dict['top_left'] = info.text.strip()
                                        elif info.tag == 'bottomright':
                                            image_descriptor_dict['bottom_right'] = info.text.strip()
                                        else:
                                            _LOG.log(50,
                                                     'Process SOIN: Unexpected tag {} encountered'.format(info.tag))
                                            raise UnexpectedXMLTag
                                    typed_descriptor_dict['descriptor'] = ImageDescriptor(**image_descriptor_dict)

                                elif descriptor.tag == 'kb_descriptor':
                                    kb_descriptor_dict = deepcopy(KB_DESCRIPTOR_REP_TEMPLATE)
                                    for info in descriptor:
                                        if info.tag == 'kbid':
                                            kb_descriptor_dict['kbid'] = info.text.strip()
                                        else:
                                            _LOG.log(50,
                                                     'Process SOIN: Unexpected tag {} encountered'.format(info.tag))
                                            raise UnexpectedXMLTag
                                    kb_descriptor_dict['dup_kbid_mapping'] = dup_kbid_mapping
                                    typed_descriptor_dict['descriptor'] = KBDescriptor(**kb_descriptor_dict)

                                else:
                                    _LOG.log(50, 'Process SOIN: Unexpected tag {} encountered'.format(descriptor.tag))
                                    print(descriptor.tag)
                                    raise UnexpectedXMLTag
                                this_td = TypedDescriptor(**typed_descriptor_dict)
                            ep_dict['typed_descriptor_list'].append(TypedDescriptor(**typed_descriptor_dict))
                soin_dict['entrypoints'].append(Entrypoint(**ep_dict))
    return SOIN(**soin_dict)

