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


##
# read output file stating relatedness between queries and claims
def read_querydoc_relatedness(filepath, qid_claimcandidates, qid_2_file, cid_2_file):
    query_relclaim = { }
    
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        # Query_Filename,Query_ID,Query_Sentence,Claim_Filename,Claim_ID,Claim_Sentence,Related or Unrelated,Score
        header = next(csv_reader)
        for row in csv_reader:
            qfilename, qid, qtext, cfilename, cid, ctext, isrel, score = row

            # santiy checks
            if qid not in qid_2_file or qid_2_file[qid][0] != qfilename or qid_2_file[qid][1] != qtext:
                print("error: mismatching info on query", qid)
                if qfilename != qid_2_file[qid][0]:
                    print("filenames:", qfilename, "vs", qid_2_file[qid][0])
                if qtext != qid_2_file[qid][1]:
                    print("text:", qtext, "vs", qid_2_file[qid][1])
                sys.exit(1)

            # santiy checks
            if cid not in cid_2_file or  cid_2_file[cid][0] != cfilename or cid_2_file[cid][1] != ctext:
                print("error: mismatching info on claim", cid)
                sys.exit(1)

            if qid not in qid_claimcandidates:
                print("error: qid has no candidates", qid)
                print(qid_claimcandidates.keys())
                sys.exit(1)

            # put qid into the dictionary so we can see if we get zero related items
            if qid not in query_relclaim: query_relclaim[qid] = [ ]
                
            # test relatedness
            if isrel == "Related" and cid in qid_claimcandidates[qid]:
                query_relclaim[qid].append(cid)

    return query_relclaim


######3
# given a queries.tsv or docclaims.tsv file, return:
# mapping from IDs to filenames and text
# mapping from IDs to topic/subtopic/list of templates
def read_query_or_docclaim_tsv(filepath):
    id_2_file = { }
    id_2_topic = { }

    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")
        # row example:
        # 'CLL0C04979A.000004.ttl', 'claim-CLL0C04979A.000004', 'Author claims masks do not trap germs',
        # 'Non-Pharmaceutical Interventions (NPIs): Masks', 'Harmful effects of wearing masks',
        # "['Wearing masks has X negative effect']"]
        #
        # column indices:
        # 0: query/claim filename
        # 1: query/claim ID
        # 2: query/claim text
        # 3: topic
        # 4 : subtopic
        # 5: template
        for row in csv_reader:
            qcfilename, qcid, qctext, qctopic, qcsubtopic, qctemplate_s = row
            if '[' in qctemplate_s and ']' in qctemplate_s:
                # templates read from AIF are lists, encoded as strings.
                qctemplate = json.loads(qctemplate_s.replace("\'", '\"'))
            else:
                # templates read from topic files are strings
                qctemplate = [ qctemplate_s] 
                
            id_2_file[qcid] = (qcfilename, qctext)
            id_2_topic[qcid] = (qctopic, qcsubtopic, qctemplate)

    return id_2_file, id_2_topic

#############33
# given a dictionary of query IDs with their topics, subtopics, and templates,
# and of claim IDs with their topics, subtopics, and templates,
# make a dictionary that maps each query ID to the claim IDs with matching topic, subtopic, and templates
def topic_matcher(query_topics, docclaim_topics, verbose = False):
    # determine candidate claims for each query
    query_candidates = { }

    for query_id, query_type in query_topics.items():
        qtopic, qsubtopic, qtemplates = query_type
        query_candidates[query_id] = [ ]
        
        for claim_id, claim_type in docclaim_topics.items():
            ctopic, csubtopic, ctemplates = claim_type
            if qtopic == ctopic and qsubtopic == csubtopic and any(qt in ctemplates for qt in qtemplates):
                query_candidates[query_id].append(claim_id)

        if len(query_candidates[query_id]) == 0 and verbose:
            print("\nWarning: no candidates found for query", query_id)
            print("Topic:", qtopic)
            print("Subtopic:", qsubtopic)
            print("Templates:", qtemplates)

    if verbose:
        # histogram of candidate counts
        count_counts = defaultdict(int)
        for query_id, candidates in query_candidates.items():
            count_counts[ len(candidates) ] += 1

        print("Sanity check: Do we have sufficient numbers of candidates, just based on topic match?")
        for count in sorted(count_counts.keys()):
            print("queries with {} candidates".format(count), ":", count_counts[count])
        print()

    return query_candidates
    
def match_everything(query_topics, docclaim_topics, verbose = False):
    # determine candidate claims for each query
    query_candidates = { }

    for query_id, query_type in query_topics.items():
        query_candidates[query_id] = [ ]
        
        for claim_id, claim_type in docclaim_topics.items():
            query_candidates[query_id].append(claim_id)

        if len(query_candidates[query_id]) == 0 and verbose:
            print("\nWarning: no candidates found for query", query_id)
            print("Topic:", qtopic)
            print("Subtopic:", qsubtopic)
            print("Templates:", qtemplates)

    if verbose:
        # histogram of candidate counts
        count_counts = defaultdict(int)
        for query_id, candidates in query_candidates.items():
            count_counts[ len(candidates) ] += 1

        print("Sanity check: Do we have sufficient numbers of candidates, just based on topic match?")
        for count in sorted(count_counts.keys()):
            print("queries with {} candidates".format(count), ":", count_counts[count])
        print()

    return query_candidates
    
###########################
def main():
    ######3
    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument('query_dir', help='Directory with preprocessed query files (cond5,6,7)')
    parser.add_argument('docclaim_dir', help="Directory with text claims in tsv")
    parser.add_argument('working_dir', help="Working directory with intermediate system results")
    parser.add_argument('condition', help="Condition5, Condition6, Condition7")
    parser.add_argument('-f', '--force', action='store_true',
                        help='If specified, overwrite existing output files without warning')
    
    
    args = parser.parse_args()

    # sanity check on condition
    if args.condition not in ["Condition5", "Condition6", "Condition7"]:
        print("Error: need a condition that is Condition5, Condition6, Condition7")
        sys.exit(1)

    ########
    # read queries tsv. 
    query_path = util.get_input_path(args.query_dir)
    query_path_cond = util.get_input_path(query_path / args.condition / "queries.tsv")

    query_filetext, query_topics = read_query_or_docclaim_tsv(str(query_path_cond))

    #######
    # read docclaims tsv

    docclaims_path = util.get_input_path(args.docclaim_dir)
    docclaims_file = util.get_input_path(docclaims_path / "docclaims.tsv")

    docclaim_filetext, docclaim_topics = read_query_or_docclaim_tsv(str(docclaims_file))

    ###
    # determine query candidates: claims with matching topic/subtopic/template
    if args.condition != "Condition7":
        query_candidates = topic_matcher(query_topics, docclaim_topics, verbose = True)
    else:
        # everything matches with everything
        query_candidates = match_everyting(query_topics, docclaim_topics, verbose = True)
    
    ########
    # read query/doc relatedness results
    working_path = util.get_input_path(args.working_dir)
    
    if args.condition != "Condition6":
        # actually do read those results and use them

        working_cond_path = util.get_input_path(working_path / args.condition, check_exist = True)
        querydoc_file = util.get_input_path(working_cond_path / "Step1_query_claim_relatedness" / "q2d_relatedness.csv")

        query_rel = read_querydoc_relatedness(querydoc_file, query_candidates, query_filetext, docclaim_filetext)

        # histogram of relatedness counts
        count_counts = defaultdict(int)
        for query_id, rel in query_rel.items():
            count_counts[ len(rel) ] += 1

        print("Sanity check: Do we have sufficient numbers of related claims?")
        for count in sorted(count_counts.keys()):
            print("queries with {} related claims".format(count), ":", count_counts[count])
        print()

    else:
        # Condition6: any candidate is related
        query_rel = query_candidates


    # ########
    # # check against yaling's file
    # yaling_path = util.get_input_path(args.yalings_file)

    # yaling_rel = defaultdict(list)

    # with open(yaling_path) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=",")
    #     # Query_Filename,Query_ID,Claim_Filename,Claim_ID
    #     header = next(csv_reader)
    #     for row in csv_reader:
    #         qfilename, qid, cfilename, cid = row

    #         if qid not in query_rel or cid not in query_rel[qid]:
    #             print("Error, entry that yaling has but I don't", qid, cid)


    #         yaling_rel[qid].append(cid)

    # for qid, rel in query_rel.items():
    #     for cid in rel:
    #         if qid not in yaling_rel or cid not in yaling_rel[qid]:
    #             print('error, entry that I have but yaling doesnt', qid, cid)
            


    #########
    # write output file: pairs of claims that have a query in common
    
    # first, invert query -> claim dict
    claim_query = defaultdict(list)
    
    for query_id, rel in query_rel.items():
        for claim_id in rel:
            claim_query[claim_id].append(query_id)

            
    # now, for each claim, determine later-named claims that have one of the same queries
    claimlist = list(claim_query.keys())
    claim_claim = { }

    for claim_index, claim_id in enumerate(claimlist):
        if claim_id not in docclaim_filetext:
            print("error, claim has no associated filename", claim_id1)
            sys.exit(1)
            
        claim_claim[ claim_id ] = [ ]
        for claim_id2 in claimlist[claim_index+1:]:
            if any(qid in claim_query[ claim_id] for qid in claim_query[claim_id2]):
                claim_claim[claim_id].append(claim_id2)


    # sanity check
    for query_id, rel in query_rel.items():
        for claim1 in rel:
            for claim2 in rel:
                if claim1 != claim2 and claim2 not in claim_claim.get(claim1, []) and claim1 not in claim_claim.get(claim2, []):
                    print("error: two claims that should be related but aren't in the list", claim1, claim2)

    # make output file
    working_cond_path = working_path / args.condition
    if not working_cond_path.exists():
        working_cond_path.mkdir(exist_ok=True, parents=True)
        
    output_path = util.get_output_dir(working_cond_path / "Step2_claimpairs" , overwrite_warning=not args.force)
    output_filename = output_path / "claim_claim.csv"

    # write csv output
    with open(str(output_filename), 'w', newline='') as csvfile:
        fieldnames = ["claim1_filename", "claim1_id", "claim1_text", "claim2_filename", "claim2_id", "claim2_text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for claim1 in claim_claim.keys():
            claim1_file, claim1_text = docclaim_filetext[claim1]
            
            for claim2 in claim_claim[claim1]:                    
                claim2_file, claim2_text = docclaim_filetext[claim2]

                writer.writerow({"claim1_filename" : claim1_file, "claim1_id" : claim1, "claim1_text": claim1_text,
                                "claim2_filename" : claim2_file, "claim2_id" : claim2, "claim2_text" : claim2_text})
                
    



if __name__ == '__main__':
    main()
