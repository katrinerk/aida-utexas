import sys
import os
import pandas
import logging
import csv
import json
import requests

from pathlib import Path
currpath = Path(os.getcwd())

sys.path.insert(0, str(currpath.parents[1]))

from aida_utexas import util
from aida_utexas.aif import JsonGraph
from collections import defaultdict
import itertools

def main():
    graph_path = "/Users/cookie/Downloads/GAIA_English.Colorado_TA2_20220211.ttl.json"
    
    json_graph = JsonGraph.from_dict(util.read_json_file(graph_path, 'JSON graph'))
    
    target_topic = "Origin of the Virus"
    
    target_subtopic = "Where the first case of COVID-19 occurred"
    
    target_claim_template = "The first case of COVID-19 occurred in location-X"
    
    #divide claims by their topic
    
    #for the claim template which only allows one X, look at the x-var, if two are not the same, should be deemed as refuting
    
    #for any claim template, with the same x-var, but epistemic status are different, true and false, should be deemed as refuting
    
    topic_claim = defaultdict(list)
    
    epistemic = set()
    claims = [ ]
    
    for claim_label, claim_entry in json_graph.each_claim():
       topic = claim_entry.topic
       claims.append(claim_entry)
       
       topic_claim[topic].append(claim_entry)
       
    refuting_claims1 = defaultdict(list)
    same_claims1 = defaultdict(list)
    refuting_claims2 = defaultdict(list)
    
    # for claim1, claim2 in itertools.combinations(claims, 2):
    #     if str(claim1.topic) == str(claim2.topic) and str(claim1.subtopic) == str(claim2.subtopic) and str(claim1.claim_template[0]) == str(claim2.claim_template[0]):
    #         xvar1 = claim1.xvar[0]
    #         xvar2 = claim2.xvar[0]
    #         xvar1_name = json_graph.node_dict[str(xvar1)].name
    #         xvar2_name = json_graph.node_dict[str(xvar2)].name
    #         if xvar1_name == xvar2_name and ((claim1.epistemic == "EpistemicTrueCertain" and claim2.epistemic == "EpistemicTrueUncertain") or (claim2.epistemic == "EpistemicTrueCertain" and claim1.epistemic == "EpistemicTrueUncertain")):
    #             refuting_claims2[claim1.claim_id].append(claim2.claim_id)
    #             refuting_claims2[claim2.claim_id].append(claim1.claim_id)
    #             print(claim1.claim_id, claim2.claim_id)
                
    # output_filename1 = "/Users/cookie/Downloads/refuting_claims_type2.tsv"
    # with open(output_filename1, 'w', newline='') as csvfile:
    #             #fieldnames = ['claim1', 'claim2']
    #             writer = csv.writer(csvfile, delimiter="\t")
    #             for claim1 in refuting_claims2.keys():
    #                 for claim2 in refuting_claims2[claim1]:                       
    #                     writer.writerow( [claim1, claim2])
    
    xvar_name = {}
    xvar_identity = {}
    
    search_api = "https://kgtk.isi.edu/api"
    similarity_api = "https://kgtk.isi.edu/similarity_api"
    similarity_set = {"topsim", "class", "jc", "complex", "transe", "text"}
    similarity_result = {}
    recorded = set()
    xvar_pool = set()
    
    for claim1, claim2 in itertools.combinations(topic_claim[target_topic], 2):
        if (claim1.claim_id, claim2.claim_id) not in recorded:
            #print("{} : {}, {}: {}\n".format(claim1.claim_id, claim1.claim_template[0], claim2.claim_id, claim2.claim_template[0]))
            if str(claim1.claim_template[0]) == target_claim_template and str(claim2.claim_template[0]) == target_claim_template:
                xvar1 = claim1.xvar[0]
                xvar2 = claim2.xvar[0]
                xvar1_identity = json_graph.node_dict[str(xvar1)].identity
                xvar2_identity = json_graph.node_dict[str(xvar2)].identity
                xvar_identity[claim1.claim_id] = (xvar1_identity)
                xvar_name[claim1.claim_id] = (json_graph.node_dict[str(xvar1)].name)
                xvar_identity[claim2.claim_id] = (xvar2_identity)
                xvar_name[claim2.claim_id] = (json_graph.node_dict[str(xvar2)].name)
            
                if xvar1_identity != xvar2_identity:
                    refuting_claims1[claim1.claim_id].append(claim2.claim_id)
                    xvar_pool.add(xvar1_identity)
                    xvar_pool.add(xvar2_identity)
                else:
                    same_claims1[claim1.claim_id].append(claim2.claim_id)
            recorded.add((claim1.claim_id, claim2.claim_id))
            recorded.add((claim2.claim_id, claim1.claim_id))

    for q1, q2 in itertools.combinations(list(xvar_pool), 2):  
        if (q1, q2) not in similarity_result.keys():
            similarity_result[(q1, q2)] = {}
            similarity_result[(q2, q1)] = {}   
        if q1[0:1] == "Q" and q2[0:1] == "Q": #avoid null identity id
            for similarity_type in similarity_set:
                query_string = similarity_api + "?q1=" + q1 + "&q2=" + q2 + "&similarity_type=" + similarity_type
                response = requests.get(query_string)
                data = response.json()
                print(data)
                similarity_result[(q1, q2)][similarity_type] = data["similarity"]
                similarity_result[(q2, q1)][similarity_type] = data["similarity"]
        else:
            for similarity_type in similarity_set:
                similarity_result[(q1, q2)][similarity_type] = "unknown"
                similarity_result[(q2, q1)][similarity_type] = "unknown"


    output_filename2 = "/Users/cookie/Downloads/refuting_claims_type1.tsv"
    with open(output_filename2, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter="\t")
                writer.writerow(["claim1", "qnode1", "name1", "claim2", "qnode2", "name2", "topsim", "class", "jc", "complex", "transe", "text"])
                for claim1 in refuting_claims1.keys():
                    for claim2 in refuting_claims1[claim1]:
                        if xvar_identity[claim1][0:1] == "Q" and xvar_identity[claim2][0:1] == "Q": 
                            if (xvar_identity[claim1], xvar_identity[claim2]) in similarity_result.keys():
                                target = similarity_result[(xvar_identity[claim1], xvar_identity[claim2])]    
                                writer.writerow( [claim1, xvar_identity[claim1], xvar_name[claim1], claim2, xvar_identity[claim2], xvar_name[claim2], target["topsim"], target["class"], target["jc"], target["complex"],target["transe"], target["text"] ])
                        else:
                            writer.writerow( [claim1, xvar_identity[claim1], xvar_name[claim1], claim2, xvar_identity[claim2], xvar_name[claim2], "0","0","0","0","0","0"])
                        
    output_filename3 = "/Users/cookie/Downloads/same_claims.tsv"
    with open(output_filename3, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter="\t")
                writer.writerow(["claim1", "qnode1", "name1", "claim2", "qnode2", "name2"])
                for claim1 in same_claims1.keys():
                    for claim2 in same_claims1[claim1]:                   
                        writer.writerow( [claim1, xvar_identity[claim1], xvar_name[claim1], claim2, xvar_identity[claim2], xvar_name[claim2]])
                        
        
        
    
    
    
    
if __name__ == '__main__':
    main()