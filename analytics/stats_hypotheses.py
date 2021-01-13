import sys
sys.path.insert(0, "/Users/kee252/Documents/Projects/AIDA/scripts/aida-utexas")

from collections import defaultdict
from typing import Dict, Iterable, List

from argparse import ArgumentParser


from aida_utexas import util
from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis.aida_hypothesis import AidaHypothesisCollection

###############################3
def hyp_stats(hyp, aobj, json_graph):
    
    aobj["stmts"].append(len(hyp.stmts))
    aobj["core_stmts"].append(len(hyp.core_stmts))
    aobj["eres"].append(len(hyp.eres()))
    aobj["core_eres"].append(len(hyp.core_eres()))
    aobj["events"].append(len(list(e for e in hyp.eres() if json_graph.is_event(e))))
    aobj["entities"].append(len(list(e for e in hyp.eres() if json_graph.is_entity(e))))
    aobj["relations"].append(len(list(e for e in hyp.eres() if json_graph.is_relation(e))))

    # count adjacent statements for core entities
    num_entity_eres = 0
    num_entities = 0
    
    for ere in hyp.core_eres():
       if json_graph.is_entity(ere):
           num_entities += 1
           for e2 in hyp.eres():
                if json_graph.is_event(e2) or json_graph.is_relation(e2):
                    if any(ere == obj_ere for stmt, pred, obj_ere in hyp.event_rel_each_arg_stmt(e2)):
                        num_entity_eres+= 1

    aobj["core_entity_evrel"].append(num_entity_eres/num_entities)

    # count adjacent statements for entities
    num_entity_eres = 0
    num_entities = 0
    
    for ere in hyp.eres():
       if json_graph.is_entity(ere):
           num_entities += 1
           for e2 in hyp.eres():
                if json_graph.is_event(e2) or json_graph.is_relation(e2):
                    if any(ere == obj_ere for stmt, pred, obj_ere in hyp.event_rel_each_arg_stmt(e2)):
                        num_entity_eres+= 1
                        
    aobj["entity_evrel"].append(num_entity_eres / num_entities)

    return aobj
    
    
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

    args = parser.parse_args()

    # read KB
    json_graph = JsonGraph.from_dict(util.read_json_file(args.graph_path, 'JSON graph'))

    # read hypotheses
    hypotheses_json = util.read_json_file(args.hypothesis_path, 'hypotheses')
    hypothesis_collection = AidaHypothesisCollection.from_json(hypotheses_json, json_graph)

    analysis_obj = defaultdict(list)
    
    for hyp in hypothesis_collection:
        analysis_obj = hyp_stats(hyp, analysis_obj, json_graph)

    # for idx in range(len(analysis_obj["stmts"])):
    #     print("-----------Hypothesis", idx, "-------")
    #     for key, val in analysis_obj.items():
    #         print(key, ":", val[idx])

    print("================ Overall =============")
    for key, val in analysis_obj.items():
        print(key, round(sum(val) / len(val), 2))
            
        
if __name__ == '__main__':
    main()
