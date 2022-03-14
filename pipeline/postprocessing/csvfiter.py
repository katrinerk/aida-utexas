import os
import sys

from pathlib import Path
import io
import pandas
import csv
from collections import defaultdict


def relevant_file_filter(csvfile_path):
    filepath = Path(csvfile_path)       
    record = pandas.read_csv(filepath)

    unique_premise = record['Query_ID'].unique()
    record.rename(columns = {'Related or Unrelated': 'Related_or_Unrelated'}, inplace=True)
    
    frecord = record[record.Related_or_Unrelated == 'Related']
    
    premise_match_hypo = {}
    hypo_match_relevant_premise = {}
    claim_ttl = {}
    for premise in unique_premise:
        premise_match_hypo[premise] = []
        for row in frecord.itertuples(index=True, name='Pandas'):
            if row.Query_ID == premise:
                if row.Claim_ID not in premise_match_hypo[premise]: #remove duplicate query-doc pairs
                    premise_match_hypo[premise].append(row.Claim_ID) # query claimid -> doc claimid
                    claim_ttl[row.Claim_ID]= row.Claim_Filename # doc claimid -> doc turtle file name
                if row.Claim_ID not in hypo_match_relevant_premise.keys():
                    hypo_match_relevant_premise[row.Claim_ID] = [] 
                if premise not in hypo_match_relevant_premise[row.Claim_ID]:
                    hypo_match_relevant_premise[row.Claim_ID].append(premise) # doc claimid -> refuting query claimids
            
    return (premise_match_hypo, hypo_match_relevant_premise, claim_ttl)

def conflict_supporting_file_filter(csvfile_path):
    filepath = Path(csvfile_path)
    
    record = pandas.read_csv(filepath)
    unique_premise = record['premise_id'].unique()
    hypo_match_supporting_premise = {}
    hypo_match_refuting_premise = {}
    claim_ttl = {}
    
    for premise in unique_premise:
        ctdrecord = record[record.adjust_label == 'contradiction']
        for row in ctdrecord.itertuples(index=True, name='Pandas'):
            if row.premise_id == premise:
                claim_ttl[row.hypo_id]= row.hypo_ttl # doc claimid -> doc turtle file name
                if row.hypo_id not in hypo_match_refuting_premise.keys():
                    hypo_match_refuting_premise[row.hypo_id] = [] 
                if premise not in hypo_match_refuting_premise[row.hypo_id]:
                    hypo_match_refuting_premise[row.hypo_id].append(premise) # doc claimid -> refuting query claimids
     
        rfrecord = record[record.adjust_label == 'entailment']
        for row in rfrecord.itertuples(index=True, name='Pandas'):
            if row.premise_id == premise:
                claim_ttl[row.hypo_id]= row.hypo_ttl # doc claimid -> doc turtle file name
                if row.hypo_id not in hypo_match_supporting_premise.keys():
                    hypo_match_supporting_premise[row.hypo_id] = []
                if premise not in hypo_match_supporting_premise[row.hypo_id]:
                    hypo_match_supporting_premise[row.hypo_id].append(premise) # doc claimid -> supporting query claimids
       
                    
    return (hypo_match_refuting_premise, hypo_match_supporting_premise, claim_ttl)  


### jy
# return relevant relations between two doc claims
def relevant_file_filter2(csvfile_path):
    filepath = Path(csvfile_path)       
    record = pandas.read_csv(filepath)
    data_top = record.head()
    unique_claim1 = record['Claim1_ID'].unique()
    frecord = record[record.Redundant_or_Independent == 'Related']
    claim_claim_related = {}
    for claim1 in unique_claim1:
        claim_claim_related[claim1] = []
        for row in frecord.itertuples(index=True, name='Pandas'):
            if row.Claim1_ID == claim1:
                if row.Claim2_ID not in claim_claim_related[claim1]:
                    claim_claim_related[claim1].append(row.Claim2_ID)
    
    return claim_claim_related

###jy
# return refuting/supporting relations between 2 doc claims
def conflict_supporting_file_filter2(csvfile_path):
    filepath = Path(csvfile_path)       
    record = pandas.read_csv(filepath)
    unique_claim1 = record['claim1_id'].unique()
    
    claim_claim_refuting = {}
    claim_claim_supporting = {}
    
    for claim1 in unique_claim1:
        ctdrecord = record[record.nli_label == 'contradiction']
        for row in ctdrecord.itertuples(index=True, name='Pandas'):
            if row.claim1_id == claim1:
                if row.claim1_id not in claim_claim_refuting.keys():
                    claim_claim_refuting[row.claim1_id] = [] 
                if row.claim2_id not in claim_claim_refuting[row.claim1_id]:
                    claim_claim_refuting[row.claim1_id].append(row.claim2_id)
     
        rfrecord = record[record.nli_label == 'entailment']
        for row in rfrecord.itertuples(index=True, name='Pandas'):
            if row.claim1_id == claim1:
                if row.claim1_id not in claim_claim_supporting.keys():
                    claim_claim_supporting[row.claim1_id] = [] 
                if row.claim2_id not in claim_claim_supporting[row.claim1_id]:
                    claim_claim_supporting[row.claim1_id].append(row.claim2_id)
                         
    return (claim_claim_refuting, claim_claim_supporting)  

def main():
    csvpath1 = "/Users/cookie/Box/AIDA/Evaluation 2022/Dry run/WORKING/TA2Colorado_Feb24/condition7/step3_claim_claim_ranking/claim_claim_redundancy.csv"
    csvpath2 = "/Users/cookie/Box/AIDA/Evaluation 2022/Dry run/WORKING/TA2Colorado_Feb24/condition7/step2_query_claim_nli/d2d_nli.csv"
    csvpath3 = "/Users/cookie/Box/AIDA/Evaluation 2022/Dry run/WORKING/TA2Colorado_Feb24/condition6/step1_query_claim_relatedness/q2d_relatedness.csv"
    csvpath4 = "/Users/cookie/Box/AIDA/Evaluation 2022/Dry run/WORKING/TA2Colorado_Feb24/condition6/step2_query_claim_nli/query_related_claims.csv"
    
    query_related = defaultdict(list)
    with open(str(csvpath4), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query_id = row["Query_ID"]
            claim_id = row["Claim_ID"]
            query_related[ query_id ].append(claim_id)

    claim_related = defaultdict(list)
    with open(str(csvpath3), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query_id = row["Query_ID"]
            claim_id = row["Claim_ID"]        
            if claim_id in query_related[ query_id ] and row["Redundant_or_Independent"] == "Related":
                claim_related[claim_id].append(query_id)
            else:
                print("not match")


    with open("/Users/cookie/Downloads/test0303/filter.tsv", 'w', newline='') as csvfile:
        fieldnames = ['Query_ID', 'Claim_ID']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for claim in claim_related.keys():
            for query in claim_related[claim]:
                writer.writerow( {"Query_ID" : query, "Claim_ID" : claim}) 
    
    '''
    match1 = relevant_file_filter2(csvpath1)
    match2, match3 = conflict_supporting_file_filter2(csvpath2)

    totalconflict = 0
    totalsupporting = 0
    for claim in match2.keys():
        totalconflict += len(match2[claim])
        print("For {}, refuting query claims are {}\n".format(claim, match2[claim]))
    print(totalconflict)
    
    for claim in match3.keys():
        totalsupporting += len(match3[claim])
        print("For {}, supporting query claims are {}\n".format(claim, match3[claim]))
    print(totalsupporting)
    
    
    totalrelevant = 0
    for claim in match1.keys():
        if match1[claim] == None:
             print("For {}, no relevant doc claims.\n".format(claim))
        else:
            print("For {}, relevant doc claims are {}\n".format(claim, match1[claim]))
        totalrelevant += len(match1[claim])
    print(totalrelevant)
    '''
    
    '''
    query_related = defaultdict(list)
    claim_related = defaultdict(list)
    queries = set()
    
    with open(str(csvpath3), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query_id = row["Query_ID"]
            claim_id = row["Claim_ID"]
            queries.add(query_id)
            if row["Redundant_or_Independent"] == "Related" and float(row["Score"]) >= 0.57:
                query_related[ query_id ].append(claim_id)
                claim_related[ claim_id ].append(query_id)  
    
    print("The number of all queries")
    print(len(queries))
    
    cnt = 0
    for query in query_related.keys():
        for claim in query_related[query]:
            cnt +=1
            print("{} : {}\n".format(query, claim))
    print("related pairs {}".format(cnt))
    '''
    

if __name__ == '__main__':
    main()             
