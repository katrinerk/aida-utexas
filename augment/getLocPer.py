import collections
import csv
import json
import os
import argparse
#import geocoder
import dill
# import pandas as pd
import sys
sys.path.append("../")
# from pipeline.training.graph_salads.gen_single_doc_graphs import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--kg_dir", type=str, default="/home/cc/data/test_file_gen", help='Directory containing origianl wiki knowledge graphs (json)')
    parser.add_argument("--kg_dir", type=str, default="/home/cc/data/js-wiki-new-ontology", help='Directory containing origianl wiki knowledge graphs (json)')
    args = parser.parse_args()
    locals().update(vars(args))

    kg_list = [os.path.join(kg_dir, file_name) for file_name in os.listdir(kg_dir)]
    geopolitic = set()
    location = set()
    people = set()

    count = 100
    #kg_list = kg_list[:1]
    print(kg_list)
    for file_name in kg_list:
        print(file_name)


        # graph = dill.load(open(file_name, 'rb'))
        # for ere in graph.eres.items():
        #     ere_obj = ere[2]
        #     print(ere)
        #     breakpoint()

        with open(file_name, "r") as read_file:
            data = json.load(read_file)
            
            # filter possible geolocation or celebrity entity
            for item in data["theGraph"].items():
                #breakpoint()
                if item[1]["type"] == "Statement" and item[1]["predicate"] == "type":
                    ere_type = item[1]["object"].split("#")[-1]
                    target_ere_id = item[1]["subject"]
                    target_ere = data["theGraph"][target_ere_id]
                    if target_ere["type"]=="Entity":
                        if "Geo" in ere_type:
                            for name in target_ere["name"]:
                                geopolitic.add(name)
                            #print(target_ere_id, target_ere)
                            # if(target_ere["type"]=="Entity"):
                            #     print("Find:",ere_type ,target_ere["name"])
                        if "Loc" in ere_type or "Place" in ere_type:
                            location.add(target_ere["name"][0])

                        if "Place" in ere_type:
                            print(ere_type ,target_ere["name"])

                        if "Person" in ere_type:
                            for name in target_ere["name"]:
                                if not name[0].islower():
                                    people.add(name)
                                    print(ere_type ,target_ere["name"])
                    
            #     for a in adj:
            #         info = a.split("/")
            #         print(info)
            #         if info[-1].startswith("LOC") or info[-1].startswith("GPE"):
            #             places.add(d)
            #         elif info[-1].startswith("PER"):
            #             people.add(d)
            #             break
        # count-=1
        # if count==0:
        #     break
    newset = location.union(geopolitic)
    cw = csv.writer(open("geolocation_full_tmp.csv",'w'))
    for val in newset:
        cw.writerow([val])

    cw = csv.writer(open("people_full.csv",'w'))
    for val in people:
        cw.writerow([val])