#  Toggles debug mode; debug mode prints large amounts of scoring information and pauses for review when a node
#  scores above a threshold specified in DEBUG_SCORE_FLOOR
DEBUG = False
DEBUG_SCORE_FLOOR = 8  # The minimum score required to trigger a pause in debug mode
BB_DEBUG = False
BB_DEBUG_FLOOR = 100
ROLE_PENALTY_DEBUG = False

#  The weights for each type of score
SCORE_WEIGHTS = {
    'name': 10,
    'descriptor': 10,
    'type': 1,
}


EP_REP_TEMPLATE = {
    "variable": None,
    "typed_descriptor_list": [],
}

TYPED_DESCRIPTOR_TEMPLATE = {
    'enttype': None,
    'descriptor': None,
}

IMAGE_DESCRIPTOR_REP_TEMPLATE = {
    "doceid": None,
    "top_left": None,
    "bottom_right": None
}

TEXT_DESCRIPTOR_REP_TEMPLATE = {
    "doceid": None,
    "start": None,
    "end": None,
}

STRING_DESCRIPTOR_REP_TEMPLATE = {
    "name_string": None,
}

KB_DESCRIPTOR_REP_TEMPLATE = {
    "kbid": None,
}

VIDEO_DESCRIPTOR_REP_TEMPLATE = {
    'doceid': None,
    'keyframe_id': None,
    'top_left': None,
    'bottom_right': None,
}

FACET_TEMPLATE = {
    "ERE": [],
    "temporal": [],
    "statements": [],
    "queryConstraints": {},
}
