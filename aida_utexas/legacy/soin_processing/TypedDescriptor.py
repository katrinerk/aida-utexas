from aida_utexas.legacy.soin_processing.templates_and_constants import DEBUG, BB_DEBUG, BB_DEBUG_FLOOR


def compute_string_overlap(observed, target):
    len_observed = observed[1] - observed[0]
    len_target = target[1] - target[0]

    # Check if they overlap at all
    if observed[1] < target[0] or target[1] < observed[0]:
        return 0

    overlap = []
    if target[0] <= observed[0]:
        overlap.append(observed[0])
    else:
        overlap.append(target[0])

    if target[1] <= observed[1]:
        overlap.append(target[1])
    else:
        overlap.append(observed[1])

    if DEBUG:
        print('overlap: ' + str(overlap))

    len_overlap = overlap[1] - overlap[0]
    overlap_target = len_overlap/len_target
    overlap_observed = len_overlap/len_observed

    return ((overlap_target + overlap_observed)/2) * 100


def compute_bounding_box_overlap(observed, target):
    # Check to see if there is overlap
    overlap = True
    if int(target['bottomright'][1]) <= int(observed['topleft'][1]):
        overlap = False
    if int(target['bottomright'][0]) <= int(observed['topleft'][0]):
        overlap = False
    if int(target['topleft'][1]) >= int(observed['bottomright'][1]):
        overlap = False
    if int(target['topleft'][0]) >= int(observed['bottomright'][0]):
        overlap = False

    if not overlap:
        return 0

    #  This handles if they intersect
    observed_l = int(observed['bottomright'][0]) - int(observed['topleft'][0])
    observed_w = int(observed['bottomright'][1]) - int(observed['topleft'][1])
    area_observed = observed_l * observed_w

    target_l = int(target['bottomright'][0]) - int(target['topleft'][0])
    target_w = int(target['bottomright'][1]) - int(target['topleft'][1])
    area_target = target_l * target_w

    overlap_coords = {
        'topleft_x': None,
        'topleft_y': None,
        'bottomright_x': None,
        'bottomright_y': None
    }

    # Determine the coordinates of the overlapping region
    if int(observed['topleft'][0]) <= int(target['topleft'][0]):
        overlap_coords['topleft_x'] = int(target['topleft'][0])
    else:
        overlap_coords['topleft_x'] = int(observed['topleft'][0])

    if int(observed['topleft'][1]) <= int(target['topleft'][1]):
        overlap_coords['topleft_y'] = int(target['topleft'][1])
    else:
        overlap_coords['topleft_y'] = int(observed['topleft'][1])

    if int(observed['bottomright'][0]) <= int(target['bottomright'][0]):
        overlap_coords['bottomright_x'] = int(observed['bottomright'][0])
    else:
        overlap_coords['bottomright_x'] = int(target['bottomright'][0])

    if int(observed['bottomright'][1]) <= int(target['bottomright'][1]):
        overlap_coords['bottomright_y'] = int(observed['bottomright'][1])
    else:
        overlap_coords['bottomright_y'] = int(target['bottomright'][1])

    overlap_l = overlap_coords['bottomright_x'] - overlap_coords['topleft_x']
    overlap_w = overlap_coords['bottomright_y'] - overlap_coords['topleft_y']
    area_overlap = overlap_l * overlap_w

    overlap_target = area_overlap / area_target
    overlap_observed = area_overlap / area_observed

    window_score = ((overlap_target + overlap_observed)/2)*100
    if BB_DEBUG and window_score > BB_DEBUG_FLOOR:
        print("OBSERVED: ")
        print("(" + str(observed['topleft'][0]) + "," + str(observed['topleft'][1]))
        print("(" + str(observed['bottomright'][0]) + "," + str(observed['bottomright'][1]))
        print("Area: " + str(area_observed) + "\n")

        print("TARGET: ")
        print("(" + str(target['topleft'][0]) + "," + str(target['topleft'][1]))
        print("(" + str(target['bottomright'][0]) + "," + str(target['bottomright'][1]))
        print("Area: " + str(area_target) + "\n")

        print("OVERLAP: ")
        print("(" + str(overlap_coords['topleft_x']) + "," + str(overlap_coords['topleft_y']) + ")")
        print("(" + str(overlap_coords['bottomright_x']) + "," + str(overlap_coords['bottomright_y']) + ")")
        print("Area: " + str(area_observed) + "\n")
        input()
    return window_score


class TypedDescriptor:
    def __init__(self, enttype, descriptor):
        self.enttype = enttype
        self.descriptor = descriptor

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'enttype': self.enttype,
            'descriptor': self.descriptor,
        }
        return str(rep)


class EntType:
    def __init__(self, typ, subtype=None, subsubtype=None):
        self.type = typ
        self.subtype = subtype
        self.subsubtype = subsubtype

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'type': self.type,
            'subtype': self.subtype,
            'subsubtype': self.subsubtype
        }
        return str(rep)

    def get_type_score(self, target):
        score = 0
        num_types = 0

        if self.type:
            num_types += 1
            if self.type.strip().lower() == target['type'].strip().lower():
                score += 100
        if self.subtype:
            num_types += 1
            if self.subtype.strip().lower() == target['subtype'].strip().lower():
                score += 100
        if self.subsubtype:
            num_types += 1
            if self.subsubtype.strip().lower() == target['subsubtype'].strip().lower():
                score += 100

        if num_types == 0:
            return 0

        if DEBUG:
            print("\n\nTYPE SCORING:")
            print("Target: " + str(target))
            print("Self: " + str(self))
            print("Score: " + str(score/num_types))
            print()
            # input()
        return score/num_types


class TextDescriptor:
    def __init__(self, doceid, start, end):
        self.descriptor_type = "Text"
        self.doceid = doceid
        self.start = start
        self.end = end

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'doceid': self.doceid,
            'start': self.start,
            'end': self.end,
        }
        return str(rep)

    def evaluate_node(self, justification_node):
        """
        This function evaluates the TextDescriptor against a justificaiton node to determine if there is a match.
        :param justification_node: AidaNode
        :return: True/False
        """
        score = 0
        justification_source = next(iter(justification_node.get("source"))).value.strip()
        if not justification_source:
            if DEBUG:
                print("TEXT DESCRIPTOR")
                print("Typed Descriptor:")
                print(str(self))
                print("\nJustification Node: ")
                justification_node.prettyprint()
                print("Score: " + str(score))
                print()

            return 0

        if justification_source == self.doceid:
            score += 10
        else:
            if DEBUG:
                print("TEXT DESCRIPTOR")
                print("Typed Descriptor:")
                print(str(self))
                print("\nJustification Node: ")
                justification_node.prettyprint()
                print("Score: " + str(score))
                print()
            return 0

        justification_start_set = justification_node.get('startOffset')
        if justification_start_set:
            justification_start_offset = int(next(iter(justification_start_set)).value)

            justification_end_set = justification_node.get('endOffsetInclusive')
            if justification_end_set:
                justification_end_offset = int(next(iter(justification_end_set)).value)
                observed = [justification_start_offset, justification_end_offset]
                target = [int(self.start), int(self.end)]
                score += compute_string_overlap(observed, target) * .9

        if DEBUG:
            print("TEXT DESCRIPTOR")
            print("Typed Descriptor:")
            print(str(self))
            print("\nJustification Node: ")
            justification_node.prettyprint()
            print("Score: " + str(score))
            print()
        # input()

        return score


class StringDescriptor:
    def __init__(self, name_string):
        self.descriptor_type = "String"
        self.name_string = name_string

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'name_string': self.name_string,
        }
        return str(rep)

    def evaluate_node(self, subject_node):
        """
        A function to evaluate teh StringDescriptor against a subject node to determine if there is a match.
        :param subject_node: AidaNode
        :return: True/False
        """
        name_set_literals = subject_node.get('hasName')
        if not name_set_literals:
            if DEBUG:
                print("STRING DESCRIPTOR")
                print("Typed Descriptor: " + str(self))
                print("Subject Node: ")
                print('[]')
                print("Score: 0")
                print()
            return 0
        names = []
        for name in name_set_literals:
            names.append(name.value.strip())

        if self.name_string in names:

            if DEBUG:
                print("STRING DESCRIPTOR")
                print("Typed Descriptor: " + str(self))
                print("Subject Node: ")
                print(names)
                print("Score: 100")
                print()
                # input()
            return 100

        if DEBUG:
            print("STRING DESCRIPTOR")
            print("Typed Descriptor: " + str(self))
            print("Subject Node: ")
            print(names)
            print("Score: 0")
            print()
            # input()
        return 0


class VideoDescriptor:
    def __init__(self, doceid, keyframe_id, top_left, bottom_right):
        self.descriptor_type = "Video"
        self.doceid = doceid
        self.keyframe_id = keyframe_id
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'doceid': self.doceid,
            'keyframe_id': self.keyframe_id,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
        }
        return str(rep)

    def evaluate_node(self, justification_node, bounding_box_node):
        score = 0

        source_set = justification_node.get('source')
        if not source_set:
            return 0

        source = next(iter(source_set)).strip()

        if source == self.doceid:
            score += 10
        else:
            return 0

        keyframe_set = justification_node.get('keyFrame')
        if not keyframe_set:
            return score

        keyframe = next(iter(keyframe_set)).strip()
        if keyframe == self.keyframe_id:
            score += 10

        if not bounding_box_node:
            return score

        bb_upper_left_x = str(next(iter(bounding_box_node.get('boundingBoxUpperLeftX'))).value).strip()
        bb_upper_left_y = str(next(iter(bounding_box_node.get('boundingBoxUpperLeftY'))).value).strip()
        bb_lower_right_x = str(next(iter(bounding_box_node.get('boundingBoxLowerRightX'))).value).strip()
        bb_lower_right_y = str(next(iter(bounding_box_node.get('boundingBoxLowerRightY'))).value).strip()

        upper_left_x, upper_left_y = self.top_left.strip().split(',')
        lower_right_x, lower_right_y = self.bottom_right.strip().split(',')

        observed = {
            'topleft': [bb_upper_left_x, bb_upper_left_y],
            'bottomright': [bb_lower_right_x, bb_lower_right_y],
        }
        target = {
            'topleft': [upper_left_x, upper_left_y],
            'bottomright': [lower_right_x, lower_right_y]
        }

        window_score = compute_bounding_box_overlap(observed, target) * .8
        score += window_score

        if DEBUG:
            print("Video Score: " + str(score))
            print("Window Score: " + str(window_score))
            justification_node.prettyprint()
            bounding_box_node.prettyprint()

        return score

class ImageDescriptor:
    def __init__(self, doceid, top_left, bottom_right):
        self.descriptor_type = "Image"
        self.doceid = doceid
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'doceid': self.doceid,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
        }
        return str(rep)

    def evaluate_node(self, justification_node, bounding_box_node):
        score = 0
        justification_source_set = justification_node.get('source')
        if not justification_source_set:
            if DEBUG:
                justification_node.prettyprint()
                bounding_box_node.prettyprint()
            return 0

        justification_source = next(iter(justification_source_set)).value.strip()
        if justification_source.strip() == self.doceid.strip():
            score += 10
        else:
            return 0

        if not bounding_box_node:
            return score

        bb_upper_left_x = str(next(iter(bounding_box_node.get('boundingBoxUpperLeftX'))).value).strip()
        bb_upper_left_y = str(next(iter(bounding_box_node.get('boundingBoxUpperLeftY'))).value).strip()
        bb_lower_right_x = str(next(iter(bounding_box_node.get('boundingBoxLowerRightX'))).value).strip()
        bb_lower_right_y = str(next(iter(bounding_box_node.get('boundingBoxLowerRightY'))).value).strip()

        upper_left_x, upper_left_y = self.top_left.strip().split(',')
        lower_right_x, lower_right_y = self.bottom_right.strip().split(',')

        observed = {
            'topleft': [bb_upper_left_x, bb_upper_left_y],
            'bottomright': [bb_lower_right_x, bb_lower_right_y],
        }
        target = {
            'topleft': [upper_left_x, upper_left_y],
            'bottomright': [lower_right_x, lower_right_y]
        }

        window_score = compute_bounding_box_overlap(observed, target) * .9

        score += window_score



        if DEBUG:
            print("Image Score: " + str(score))
            print("Window Score: " + str(window_score))
            justification_node.prettyprint()
            bounding_box_node.prettyprint()

        return score


class KBDescriptor:
    def __init__(self, kbid, dup_kbid_mapping=None):
        self.descriptor_type = "KB"
        self.kbid = [kbid]
        if kbid in dup_kbid_mapping:
            self.kbid.append(dup_kbid_mapping[kbid])

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = {
            'kbid': self.kbid,
        }
        return str(rep)

    def evaluate_node(self, link_node):
        link_value_set = link_node.get('linkTarget')
        if not link_value_set:
            return 0

        link_value = next(iter(link_value_set)).value.strip()

        if link_value in self.kbid:
            return 100

        return 0
