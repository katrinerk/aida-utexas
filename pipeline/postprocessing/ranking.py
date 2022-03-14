import sys
import os


# find relative path_jy
from pathlib import Path
currpath = Path(os.getcwd())

sys.path.insert(0, str(currpath.parents[1]))


import io
import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
import csv
import json

from aida_utexas import util
#from aida_utexas import claimutil

#########3
# make a ranking for a given query
def make_ranking(query_id, query_claim_score, claim_claim_score, query_2_text, claim_2_text):
    
    # determine the highest-ranked claim
    query_claims = query_claim_score[query_id].keys()

    if len(query_claims) == 0:
        return [], {}
    
    max_claim = sorted(query_claims, key = lambda claim:query_claim_score[query_id][claim], reverse = True)[0]

    # print("\nQuery", query_2_text[query_id])
    # print("\nQuery relatedness scores:\n")
    # for c in sorted(query_claims, key = lambda claim:query_claim_score[query_id][claim], reverse = True):
    #     print(c, query_claim_score[query_id][c], claim_2_text[c])

    # print()
        
    claims_ranked = [ max_claim ]
    claim_score = { }
    claim_score[ max_claim] = query_claim_score[query_id][max_claim]
    
    while len(claims_ranked) < len(query_claims):
        summed_score = { }

        # for each claim that is not yet ranked:
        # determine summed score with all already ranked claims
        for claim1 in query_claims:
            if claim1 in claims_ranked: continue

            summed_score[claim1] = 0
            for claim2 in claims_ranked:
                
                if (claim1, claim2) not in claim_claim_score:
                    print("Error, missing score for", claim1, claim2)
                    sys.exit(0)

                summed_score[claim1] += claim_claim_score[ (claim1, claim2) ]

        # include the claim with minimum summed score, minimum relatedness to other claims
        min_claim = sorted(summed_score.keys(), key = lambda claim:summed_score[claim])[0]

        # print("\nreranked:")
        # for c in sorted(summed_score.keys(), key = lambda claim:summed_score[claim]):
        #     print(c, summed_score[c], claim_2_text[c])
        
        # print()
            

        claims_ranked.append(min_claim)
        claim_score[min_claim] = summed_score[min_claim]

    return claims_ranked, claim_score
    
###########################
def main():
    ######3
    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument('working_dir', help="Working directory with intermediate system results")
    parser.add_argument('run_id', help="run ID, same as subdirectory of Working")
    parser.add_argument('condition', help="Condition5, Condition6, Condition7")
    parser.add_argument('-f', '--force', action='store_true',
                        help='If specified, overwrite existing output files without warning')
    
    
    args = parser.parse_args()

    # sanity check on condition
    if args.condition not in ["Condition5", "Condition6", "Condition7"]:
        print("Error: need a condition that is Condition5, Condition6, Condition7")
        sys.exit(1)

    working_mainpath = util.get_input_path(args.working_dir)
    working_path = util.get_input_path(working_mainpath / args.run_id / args.condition)        
    

    # for each query, determine the list of related claims that match in topic/subtopic/template
    # this is in [working_path] / step3_claim_claim_ranking / query_claim_matching.csv
    query_related = defaultdict(list)

    filename= util.get_input_path(working_path / "step3_claim_claim_ranking" / "query_claim_matching.csv")
    
    with open(str(filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query_id = row["Query_ID"]
            claim_id = row["Claim_ID"]
            query_related[ query_id ].append(claim_id)

    # if condition5, read supporting/refuting/related relations of claims to queries

    if args.condition == "Condition5":
        query_claim_relation = { }

        # claims that are related and matching: located in [working_path]/step2_query_claim_nli/q2d_nli.csv
        queryclaim_rel_filename = util.get_input_path(working_path / "step2_query_claim_nli" / "q2d_nli.csv")
    
        with open(str(queryclaim_rel_filename), newline='') as csvfile:
            # first row has header, so we don't need to give fieldnames
            reader = csv.DictReader(csvfile)
            for row in reader:
                query_id = row["Query_ID"]
                claim_id = row["Claim_ID"]
                nli_label = row["nli_label"]
                nist_label = {"neutral" : "related", "contradiction" : "refuting", "entailment" : "supporting"}[nli_label]

                query_claim_relation[ (query_id, claim_id) ] = nist_label
    else:
        query_claim_relation = { }

    # read relatedness ratings for claim pairs
    # this is in [working_path] / step3_claim_claim_ranking / claim_claim_redundancy.csv

    claim_claim_score = { }

    filename= util.get_input_path(working_path / "step3_claim_claim_ranking" / "claim_claim_redundancy.csv")
    
    with open(str(filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            claim1 = row["Claim1_ID"]
            claim2 = row["Claim2_ID"]
            score = row["Score"]

            claim_claim_score[ (claim1, claim2) ] = float(score)
            claim_claim_score[ (claim2, claim1) ] = float(score)

    # read relatedness ratings between query and claims
    # this is in [working_path] / step1_query_claim_relatedness / q2d_relatedness.csv
    query_claim_score = defaultdict(dict)
    query_2_text = { }
    claim_2_text = { }

    filename= util.get_input_path(working_path / "step1_query_claim_relatedness" / "q2d_relatedness.csv")
    
    with open(str(filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query_id = row["Query_ID"]
            query_text = row["Query_Sentence"]
            claim_id = row["Claim_ID"]
            claim_text = row["Claim_Sentence"]
            score = row["Score"]

            if claim_id in query_related[ query_id ]:
                query_claim_score[ query_id][claim_id]  = float(score)

            query_2_text[ query_id ] = query_text
            claim_2_text[ claim_id ] = claim_text
        
    # for each query, produce a ranking file named
    # QueryID.ranking.tsv.
    # Fields:
    # Condition5: Query_ID Claim_ID Rank Relation_to_Query
    # Condition6: Query_ID Claim_ID Rank
    # Condition7: Query_ID Claim_ID Rank

    # output is to: [working_path] / step4_ranking


    output_path = util.get_output_dir("/Users/cookie/Downloads/step4_ranking2" , overwrite_warning=not args.force)

    for query_id in query_related.keys():
        # make a ranking
        ranked_claims, claim_scores = make_ranking(query_id, query_claim_score, claim_claim_score, query_2_text, claim_2_text)
        
        # print("Sanity check")
        # print("Query", query_id, query_2_text[query_id], "\n")
        # for claim in ranked_claims:
        #     print(claim, claim_scores[claim], claim_2_text[claim])

        # and write it to a file
        output_filename = output_path / (query_id + ".ranking.tsv")

        # Condition5: write including relation to query
        if args.condition == "Condition5":

            with open(output_filename, 'w', newline='') as csvfile:
                #fieldnames = ['Query_ID', 'Claim_ID', 'Rank', 'Relation_to_Query']
                writer = csv.writer(csvfile, delimiter="\t")

                #writer.writeheader()

                for rank, claim_id in enumerate(ranked_claims):
                    rel = query_claim_relation.get( (query_id, claim_id), "related")                        
                    writer.writerow( [ query_id, claim_id, rank + 1, rel ])

        # Conditions 6, 7: write without relation to query
        else:
            with open(output_filename, 'w', newline='') as csvfile:
                fieldnames = ['Query_ID', 'Claim_ID', 'Rank']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")

                writer.writeheader()
                for rank, claim_id in enumerate(ranked_claims):
                    writer.writerow( { "Query_ID" : query_id, "Claim_ID" : claim_id, "Rank" : rank + 1})



if __name__ == '__main__':
    main()
