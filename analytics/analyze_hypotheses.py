import sys
sys.path.insert(0, "/Users/kee252/Documents/Projects/AIDA/scripts/aida-utexas")

from collections import defaultdict
from typing import Dict, Iterable, List

from argparse import ArgumentParser


from aida_utexas import util
from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis.aida_hypothesis import AidaHypothesisCollection

##########################################################3
#######################3
# collecting and joining information across hypotheses

###########
# make a dictionary
# Event/rel name .role name -> filler name -> [filler ID]
# for core events and relations (relations that contain at least one statement
# that the SoIN asks about)
# NOTE: originally, only considered statements that the SoIN asks about,
# but we also need to take into account other roles of those same events/relations
# for a single hypothesis
def make_core_rolefiller_dict_forhyp(hyp, json_graph, retv):
    for ere in hyp.core_eres():
        if json_graph.is_event(ere) or json_graph.is_relation(ere):
            for stmt, pred, obj in hyp.event_rel_each_arg_stmt(ere):
                
                # keep this if the obj is an entity
                if json_graph.is_entity(obj):
                    if pred not in retv:
                        retv[ pred] = { }
                    # construct name of entity
                    name = ",".join(json_graph.ere_names(obj)[:3])
                    if len(name) == 0:
                        name = ",".join(list(json_graph.ere_types(obj)))
                    if name not in retv[pred]:
                        retv[pred][name] = set()
                        
                    # store pred -> name -> entity
                    retv[pred][name].add(obj)
    return retv

# make dictionary of core role fillers for all hypotheses in collection
def make_core_rolefiller_dict(hypothesis_collection, json_graph):
    retv = { }

    # instead, let's consider core events and relations
    for hyp in hypothesis_collection:
        retv = make_core_rolefiller_dict_forhyp(hyp, json_graph, retv)
                        
    return retv

##############3
def collect_entity_info(json_graph, hypotheses, input_ere, roles_ontology):
    retv = { }

    if not json_graph.is_entity(input_ere):
        return retv
    
    for hyp in hypotheses:
        for ere in hyp.eres():
            if json_graph.is_event(ere) or json_graph.is_relation(ere):
                roles_filled = set(pred for stmt, pred, object_ere in hyp.event_rel_each_arg_stmt(ere) if input_ere == object_ere)
                if len(roles_filled) > 0:
                    this_event_dict = collect_even_rel(ere, hyp, json_graph, roles_ontology)

                    if ere in retv:
                        retv[ere]["type"] = retv[ere]["type"].union(this_event_dict.get("type", set()))

                        for role in this_event_dict:
                            if role == "type":
                                continue
                            if role in retv[ere]:
                                retv[ere][role] = retv[ere][role].union(this_event_dict[role])
                            else:
                                retv[ere][role] = this_event_dict[role]


                        retv[ere]["EntityRole"] = retv[ere]["EntityRole"].union(roles_filled)
                    else:
                        retv[ ere] = this_event_dict
                        retv[ere]["EntityRole"] = roles_filled
    return retv
            
##
# collect output for an event or relation
def collect_even_rel(ere_label, hyp, json_graph, roles_ontology):
    retv = { }
    
    if not (json_graph.is_event(ere_label) or json_graph.is_relation(ere_label)):
        return retv

    event_rel_type = set(hyp.ere_types(ere_label))

    event_rel_roles = defaultdict(set)
    for pred_label, arg_label in hyp.event_rel_each_arg(ere_label):
        event_rel_roles[pred_label].add(arg_label)

    if not event_rel_roles:
        return {}

    if len(event_rel_type) == 0:
        event_rel_type = set([list(event_rel_roles.keys())[0].rsplit('_', maxsplit=1)[0]])

    retv["type"] = event_rel_type
    for onetype in event_rel_type:
        for role_label in roles_ontology[onetype].values():
            pred_label = onetype  + '_' + role_label
            if pred_label in event_rel_roles:
                retv[role_label] = set(entity_to_str(json_graph, arg_label) for arg_label in event_rel_roles[pred_label])

    return retv

    

#######################################################3
#######################3
# Hypothesis filtering for the purpose of analysis in this tool

####
# filter hypotheses by question IDs:
# only retain hypotheses that have some question ID
# that contains the given letter sequence
def filter_hypotheses_by_question(hypothesis_collection, questionid):
    retain_hyp = [ ]

    for hyp in hypothesis_collection:
        if any(questionid in hyp_questionid for hyp_questionid in hyp.questionIDs):
            retain_hyp.append(hyp)

    return AidaHypothesisCollection(retain_hyp)

######33
# filter hypotheses by entry points:
# only retain hypotheses that have this particular filler for this particular core role
# in their list of core ere labels
def filter_hypotheses_by_entrypoints(hypothesis_collection, json_graph, core_role, core_filler):
    retain_hyp = [ ]

    for hyp in hypothesis_collection:
        roledict = make_core_rolefiller_dict_forhyp(hyp, json_graph, {})
        if core_role in roledict and any(core_filler in fillers for name, fillers in roledict[core_role].items()):
            retain_hyp.append(hyp)

    return AidaHypothesisCollection(retain_hyp)

#######################################################
#######################3
# Turning hypothesis-specific info into output strings

###############
# print out info about an ERE in a hypothesis:
# for an event or relation, that event or relation.
# for an entity, all events or relations that talk about this entity
# again, stuff in the hyptohesis only
def str_ere_info(json_graph, hyp, input_ere, roles_ontology):
    retv = ""
    
    if input_ere in hyp.eres():
        retv += "=============\n"
        if json_graph.is_event(input_ere) or json_graph.is_relation(input_ere):
            retv += event_rel_to_str(input_ere, hyp, json_graph, roles_ontology) + "\n"
        else:
            for ere in hyp.eres():
                if json_graph.is_event(ere) or json_graph.is_relation(ere):
                    roles_filled = [pred for stmt, pred, object_ere in hyp.event_rel_each_arg_stmt(ere) if input_ere == object_ere]
                    if len(roles_filled) > 0:
                        retv += "\nERE is " + ",".join(roles_filled) +  " in:\n"
                        retv += event_rel_to_str(ere, hyp, json_graph, roles_ontology) + "\n"
    return retv

###########
# entity name
def entity_to_str(json_graph, ere_label):
    names = json_graph.ere_names(ere_label)
    if names is None or names == []:
        name = "[unknown]"
    else:
        english_names = json_graph.english_names(names)
        if len(english_names) > 0:
            name =  min(english_names, key=lambda n: len(n))
        else:
            name = min(names, key=lambda n: len(n))

    if name == '[unknown]':
        for type_label in json_graph.ere_types(ere_label):
            name = type_label
            break
    return name + " " + ere_label[-15:]


# human-readable output for an event or relation
def event_rel_to_str(ere_label, hyp, json_graph, roles_ontology):
    if not (json_graph.is_event(ere_label) or json_graph.is_relation(ere_label)):
        return ''

    event_rel_type = None
    for type_label in hyp.ere_types(ere_label):
        event_rel_type = type_label
        break

    event_rel_roles = defaultdict(set)
    for pred_label, arg_label in hyp.event_rel_each_arg(ere_label):
        event_rel_roles[pred_label].add(arg_label)

    if not event_rel_roles:
        return ''

    if event_rel_type is None:
        event_rel_type = list(event_rel_roles.keys())[0].rsplit('_', maxsplit=1)[0]

    result = event_rel_type
    for role_label in roles_ontology[event_rel_type].values():
        result += '\n    ' + role_label + ': '
        pred_label = event_rel_type + '_' + role_label
        if pred_label in event_rel_roles:
            result += '\n\t'.join(entity_to_str(json_graph, arg_label)
                                for arg_label in event_rel_roles[pred_label])

    return result


########
def ere_to_str(ere_label, hyp, json_graph, roles_ontology):
    if json_graph.is_event(ere_label) or json_graph.is_relation(ere_label):
        return event_rel_to_str(ere_label, hyp, json_graph, roles_ontology)
    elif json_graph.is_entity(ere_label):
        return entity_to_str(json_graph, ere_label)
    else:
        return ''

#########
# hypothesis to string
def hyp_to_str(hyp, json_graph, roles_ontology):
    result = ''

    core_eres = hyp.core_eres()

    # start with core EREs
    result += "========= Core events and relations =======\n\n"
    for ere_label in core_eres:
        if json_graph.is_event(ere_label) or json_graph.is_relation(ere_label):
            ere_str = ere_to_str(ere_label, hyp, json_graph, roles_ontology)
            if ere_str != '':
                result += ere_str + '\n\n'

    # for each entity in the hypothesis, print out all related
    # events and relations
    result += "\n\n============= Dramatis personae =============\n\n"
    for ere_label in hyp.eres():
        if json_graph.is_entity(ere_label):
            result += "\n-------------- entity " + ere_label[-15:] + "-------\n"
            result += str_ere_info(json_graph, hyp, ere_label, roles_ontology)
    
    return result

#######################################################
#######################3
# print information collected across multiple hypotheses
def print_collected_entity_info(input_ere, json_graph, collected_entity_info):
  for eid, entry in collected_entity_info.items():
    print("\nEntity is " + ",".join(entry["EntityRole"]) + " in:")
    print(",".join(entry["type"]), eid[-15:])
    for role_label in entry.keys():
        if role_label == "type" or role_label == "EntityRole": continue
        print(role_label, ":")
        for arg_label in entry[role_label]:
            print("\t", arg_label)
    print()


#######################################################
#######################3
# stringify and print informatoin for the whole graph,
# not a single hypothesis

#######3
# print out info about an event/relation in the whole graph:
# for an event or relation, that event or relation.
def print_eventrel_info_wholegraph(json_graph, ere_label, roles_ontology):
    if not (json_graph.is_event(ere_label) or json_graph.is_relation(ere_label)):
        return

    event_rel_type = None
    for type_label in json_graph.ere_types(ere_label):
        event_rel_type = type_label
        break

    event_rel_roles = defaultdict(set)
    for stmt in json_graph.each_ere_adjacent_stmt(ere_label):
        pred_label = json_graph.stmt_predicate(stmt)
        arg_label = json_graph.stmt_object(stmt)
        if arg_label == ere_label:
            continue
        event_rel_roles[pred_label].add(arg_label)

    if not event_rel_roles:
            return

    if event_rel_type is None:
            event_rel_type = list(event_rel_roles.keys())[0].rsplit('_', maxsplit=1)[0]

    result = event_rel_type
    for role_label in roles_ontology[event_rel_type].values():
        result += '\n    ' + role_label + ': '
        pred_label = event_rel_type + '_' + role_label
        if pred_label in event_rel_roles:
            result += '\n\t\t'.join(entity_to_str(json_graph, arg_label) for arg_label in event_rel_roles[pred_label])

    print(result)

#######3
# print out info about an event/relation in the whole graph:
# for an entity, print all events or relations that talk about this entity
def print_entity_info_wholegraph(json_graph, ere_label, roles_ontology):
    if not (json_graph.is_entity(ere_label)):
        return

    for stmt in json_graph.each_ere_adjacent_stmt(ere_label):
        ev_label = json_graph.stmt_subject(stmt)
        pred_label = json_graph.stmt_predicate(stmt)
        arg_label = json_graph.stmt_object(stmt)
        if arg_label == ere_label:
            print("\nERE is", pred_label, "in:")
            print_eventrel_info_wholegraph(json_graph, ev_label, roles_ontology)


#######################################################
#######################3
# main functions matching the main options in the menu

#####3
# print the core of a hypothesis:
# roles of core events and relations, with all fillers given in
# any of the hypotheses in the collection
def show_core(json_graph, hypothesis_collection):
    # chart possible perspectives, and output that
    core_rolefillers = make_core_rolefiller_dict(hypothesis_collection, json_graph)

    for rolelabel, nameobjs in core_rolefillers.items():
        print(rolelabel)
        for name, objs in nameobjs.items():
            print("\t", name)
            for obj in objs:
                print("\t\t", obj)


#####3
# for an ERE, show all info that a given hypothesis collection has about it
def show_ere(json_graph, hypothesis_collection, roles_ontology):
    input_ere = input("ERE ID to inspect: ")

    collected_entity_info = collect_entity_info(json_graph, hypothesis_collection, input_ere, roles_ontology)
    print_collected_entity_info(input_ere, json_graph, collected_entity_info)
    
    # # iterate through hypotheses, say what they say about this entity
    
    # for hyp in hypothesis_collection:
    #     print(str_ere_info(json_graph, hyp, input_ere, roles_ontology))

#####3
# for a core role, show all info that a hypothesis collection
# has about any of the fillers of that core role
def show_rolefiller(json_graph, hypothesis_collection, roles_ontology):
    input_role = input("Role label to inspect: ")

    core_rolefillers = make_core_rolefiller_dict(hypothesis_collection, json_graph)
    if input_role in core_rolefillers:
        for name, eres in core_rolefillers[input_role].items():
            for ere in eres:
                print("-------------- ERE named", name, "-------")
                collected_entity_info = collect_entity_info(json_graph, hypothesis_collection, ere, roles_ontology)
                print_collected_entity_info(ere, json_graph, collected_entity_info)
                #  for hyp in hypothesis_collection:
                #      print(str_ere_info(json_graph, hyp, ere, roles_ontology))
            
            
# for an ere, show its surroundings in the json graph,
# independently of hypotheses
def show_ere_graphenv(json_graph, roles_ontology):
    input_ere = input("ERE ID to inspect: ")

     # event or relation: just print it, with all its arguments
    if json_graph.is_event(input_ere) or json_graph.is_relation(input_ere):
        print("ERE is event/relation:")
        print_eventrel_info_wholegraph(json_graph, input_ere, roles_ontology)
            
    # entity: print all adjacent events or relations
    else:
        print("ERE is entity occurring as follows:")
        print_entity_info_wholegraph(json_graph, input_ere, roles_ontology)

# for a role, show its surroundings in the json graph,
# independently of hypotheses
def show_role_graphenv(json_graph, hypothesis_collection, roles_ontology):
    input_role = input("Role label to inspect: ")

    core_rolefillers = make_core_rolefiller_dict(hypothesis_collection, json_graph)
    if input_role in core_rolefillers:
        for name, eres in core_rolefillers[input_role].items():
            for ere in eres:
                print("-------------- ERE named", name, "-------")
                print_entity_info_wholegraph(json_graph, ere, roles_ontology)

# print out hypotheses
def print_hypotheses(json_graph, hypothesis_collection, roles_ontology):
    outdir_raw = input("Directory to write to: ")

    output_dir = util.get_output_dir(outdir_raw, overwrite_warning=True)

    for idx, hypothesis in enumerate(hypothesis_collection.hypotheses):
        output_path = output_dir / 'hypothesis-{:0>3d}.txt'.format(idx)
        with open(str(output_path), "w", encoding="utf-8") as fout:
            print(hyp_to_str(hypothesis, json_graph, roles_ontology), file=fout)
            
###############################
# main:
# read KB, read hypotheses
# filter hypotheses by question IDs, and optionally entry points
# list fillers for query statement roles
def main():
    parser = ArgumentParser()
    # required positional
    parser.add_argument('graph_path', help='path to the graph JSON file')
    parser.add_argument('hypothesis_path', help='path to the JSON file with hypotheses')
    parser.add_argument("roles_ontology_path", help="path to roles ontology")

    args = parser.parse_args()

    print("Reading in data...")
    
    # read KB
    json_graph = JsonGraph.from_dict(util.read_json_file(args.graph_path, 'JSON graph'))

    # read hypotheses
    hypotheses_json = util.read_json_file(args.hypothesis_path, 'hypotheses')
    hypothesis_collection = AidaHypothesisCollection.from_json(hypotheses_json, json_graph)

    # read roles ontology
    roles_ontology = util.read_json_file(args.roles_ontology_path, 'roles ontology')

    # determine all question IDs
    questionIDs = set()
    for h in hypothesis_collection:
        questionIDs.update(h.questionIDs)

    choice = question_id = restrict_core_role = restrict_core_ere = None
    while choice != "x":
        # determine core choice
        print("question IDs:", ", ".join(questionIDs))
        print("Choose from:")
        print("c: core hypothesis display")
        print("e: show events/relations connected to an ere")
        print("r: show events/relations connected to a role filler")
        print("se: survey context of an ERE independent of hypotheses")
        print("sr: survey context of a role filler independent of hypotheses")
        print("p: print hypotheses for a particular question ID")
        print("R: restrict hypotheses to be considered going forward, for the rest of the run")
        print("x: exit")
        
        choice = input()

        # determine additional restrictions on hypotheses to consider
        if choice in ["c", "e", "r", "p"]:
            question_id = input("Question ID: ")

            # filter hypotheses by question ID
            this_hypothesis_collection = filter_hypotheses_by_question(hypothesis_collection, question_id)

            # additionally filter by a core role filler?
            restrict_core_role = input("Optional core role to restrict: ")
            if restrict_core_role != "":
                restrict_core_ere = input("Value to restrict the core role to (ERE ID): ")

                this_hypothesis_collection = filter_hypotheses_by_entrypoints(this_hypothesis_collection, json_graph, restrict_core_role,
                                                                                  restrict_core_ere)


        # execute choice
        if choice == "c":
            show_core(json_graph, this_hypothesis_collection)
        elif choice == "e":
            show_ere(json_graph, this_hypothesis_collection, roles_ontology)
        elif choice == "r":
            show_rolefiller(json_graph, this_hypothesis_collection, roles_ontology)
        elif choice == "se":
            show_ere_graphenv(json_graph, roles_ontology)
        elif choice == "sr":
            show_role_graphenv(json_graph, this_hypothesis_collection, roles_ontology)
        elif choice == "R":
            restrict_core_role = input("Core role to restrict: ")
            restrict_core_ere = input("Value to restrict the core role to (ERE ID): ")

            hypothesis_collection = filter_hypotheses_by_entrypoints(hypothesis_collection, json_graph, restrict_core_role, restrict_core_ere )               
        elif choice == "p":
            print_hypotheses(json_graph, hypothesis_collection, roles_ontology)
        
if __name__ == '__main__':
    main()
