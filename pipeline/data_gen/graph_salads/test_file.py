import dill

event_types_map = dill.load(open('/home/atomko/test_event_entity_map_out/event_types.p', 'rb'))
entity_types_map = dill.load(open('/home/atomko/test_event_entity_map_out/entity_types.p', 'rb'))
event_names_map = dill.load(open('/home/atomko/test_event_entity_map_out/event_names.p', 'rb'))

print(event_types_map)