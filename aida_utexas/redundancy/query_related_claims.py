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

######3
# given a queries.tsv or docclaims.tsv file, return:
# mapping from IDs to filenames and text
# mapping from IDs to topic/subtopic/list of templates
def read_query_or_docclaim_tsv(filepath, extend_query_ids = False):
    id_2_file = { }
    id_2_topic = { }

    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t", quoting=csv.QUOTE_NONE)
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
                # template from AIF: list encoded as string
                qctemplate = json.loads(qctemplate_s.replace("\'", '\"'))
            else:
                qctemplate = [ qctemplate_s ]

            # extend query ID? if so, add query text to it
            if extend_query_ids:
                qcid = (qcid, qctext)
                
            id_2_file[qcid] = (qcfilename, qctext)
            id_2_topic[qcid] = (qctopic, qcsubtopic, qctemplate)

    return id_2_file, id_2_topic


##
# read file stating relatedness between queries and claims
def read_querydoc_relatedness(filepath, qid_claimcandidates, qid_2_file, cid_2_file, threshold = None, extend_query_ids = False):
    query_relclaim = { }
    
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        # Query_Filename,Query_ID,Query_Sentence,Claim_Filename,Claim_ID,Claim_Sentence,Related or Unrelated,Score
        header = next(csv_reader)
        for row in csv_reader:
            qfilename, qid, qtext, cfilename, cid, ctext, isrel, score = row

            if extend_query_ids:
                qid = (qid, qtext)

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
                if cid not in cid_2_file: print("claim ID missing file info")
                if cid_2_file[cid][0] != cfilename:
                    print("mismatching file info", cid_2_file[cid][0], cfilename)
                if cid_2_file[cid][1] != ctext:
                    print("mismatching claim text", cid_2_file[cid][1], ctext)
                sys.exit(1)

            if qid not in qid_claimcandidates:
                print("error: qid has no candidates", qid)
                print(qid_claimcandidates.keys())
                sys.exit(1)

            # put qid into the dictionary so we can see if we get zero related items
            if qid not in query_relclaim: query_relclaim[qid] = [ ]
                
            # test relatedness
            if (threshold is None and isrel == "Related") or (threshold and float(score) >= threshold):
                if cid in qid_claimcandidates[qid]:
                    query_relclaim[qid].append(cid)

    return query_relclaim




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

#############33
# given a dictionary of topic IDs with topics
# and of claim IDs with their topics, subtopics, and templates,
# make a dictionary that maps each topic ID to the claim IDs with matching topic, subtopic, and templates
def topic_matcher_nosubtopics(query_topics, docclaim_topics, verbose = False):
    # determine candidate claims for each query
    query_candidates = { }

    for query_id, query_type in query_topics.items():
        qtopic, _, _ = query_type
        query_candidates[query_id] = [ ]
        
        for claim_id, claim_type in docclaim_topics.items():
            ctopic, _, _ = claim_type
            if qtopic == ctopic:
                query_candidates[query_id].append(claim_id)

        if len(query_candidates[query_id]) == 0 and verbose:
            print("\nWarning: no candidates found for query", query_id)
            print("Topic:", qtopic)

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


##################3
# determine claim pairs that have a query in common.
# returns a dictionary: claim => list of other claims
# each claim lists as "other claims" only ones that come later in the overall list of claims.

def claimpairs_that_have_a_query_in_common(query_rel, query_filetext, docclaim_filetext, generalize_query_ids = False):
    # first, invert query -> claim dict
    claim_query = defaultdict(list)

    
    for query_id, rel in query_rel.items():
        if generalize_query_ids:
            if len(query_id) != 2:
                print("Error, was expecting query ID to consist of query ID and text for condition 6, got:", query_id)
                sys.exit(1)

            actual_query_id, qtext = query_id
            query_id = actual_query_id

        for claim_id in rel:
            if query_id not in claim_query[claim_id]:
                claim_query[claim_id].append(query_id)


    # print("HIER1 claim_L0C04ATMB_0", "claim_L0C04ATMB_0" in claim_query)
    # print("HIER1 claim_L0C04CA4U_7", "claim_L0C04CA4U_7" in claim_query)
            
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
    
    return claim_claim

########
# write a file with claim-claim pairs that have a query in common
def write_claimclaim_file(claim_claim, output_filename, docclaim_filetext):
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
                

    

###########################
def main():
    ######3
    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument('query_dir', help='Directory with preprocessed query files (cond5,6,7)')
    parser.add_argument('docclaim_dir', help="Directory with text claims in tsv")
    parser.add_argument('working_dir', help="Working directory with intermediate system results")
    parser.add_argument('run_id', help="run ID, same as subdirectory of Working")
    parser.add_argument('condition', help="condition5, condition6, condition7")
    parser.add_argument('-f', '--force', action='store_true',
                        help='If specified, overwrite existing output files without warning')
    parser.add_argument('-t', '--threshold', help="threshold for counting a claim as related", type = float)
    
    
    args = parser.parse_args()

    # sanity check on condition
    if args.condition not in ["condition5", "condition6", "condition7"]:
        print("Error: need a condition that is condition5, condition6, condition7")
        sys.exit(1)

    ########
    # read queries tsv. 
    query_path = util.get_input_path(args.query_dir)
    query_path_cond = util.get_input_path(query_path / args.condition / "queries.tsv")

    # in condition 6, special handling of query IDs
    query_filetext, query_topics = read_query_or_docclaim_tsv(str(query_path_cond), extend_query_ids = (args.condition == "condition6"))

    #######
    # read docclaims tsv

    docclaims_path = util.get_input_path(args.docclaim_dir)
    docclaims_file = util.get_input_path(docclaims_path / "docclaims.tsv")

    docclaim_filetext, docclaim_topics = read_query_or_docclaim_tsv(str(docclaims_file))

    ###
    # determine query candidates: claims with matching topic/subtopic/template
    if args.condition != "condition7":
        query_candidates = topic_matcher(query_topics, docclaim_topics, verbose = True)
    else:
        # in condition 7, everything matches with everything because we don't have topics
        # query_candidates = match_everything(query_topics, docclaim_topics, verbose = True)
        query_candidates = topic_matcher_nosubtopics(query_topics, docclaim_topics, verbose = True)

    
    ########
    # read query/doc relatedness results
    working_mainpath = util.get_input_path(args.working_dir)
    working_path = util.get_input_path(working_mainpath / args.run_id / args.condition)
    
    querydoc_file = util.get_input_path(working_path / "step1_query_claim_relatedness" / "q2d_relatedness.csv")

    query_rel = read_querydoc_relatedness(querydoc_file, query_candidates, query_filetext, docclaim_filetext,
                                          threshold = args.threshold,
                                          extend_query_ids = (args.condition == "condition6"))

    # histogram of relatedness counts
    count_counts = defaultdict(int)
    for query_id, rel in query_rel.items():
        count_counts[ len(rel) ] += 1

    print("Sanity check: Do we have sufficient numbers of related claims?")
    for count in sorted(count_counts.keys()):
        print("queries with {} related claims".format(count), ":", count_counts[count])
    print()



    #########
    # write output file: pairs of claims that have a query in common
    
    claim_claim = claimpairs_that_have_a_query_in_common(query_rel, query_filetext, docclaim_filetext)
    
    # make output file
    #  working_cond_path = working_path / args.condition
    # if not working_cond_path.exists():
    #     working_cond_path.mkdir(exist_ok=True, parents=True)

    output_path = working_path / "step2_query_claim_nli"
    if not output_path.exists():
        output_path.mkdir(exist_ok = True, parents = True)

    # output_path = util.get_output_dir(working_path / "step2_query_claim_nli" , overwrite_warning=not args.force)
    output_filename = output_path / "claim_claim.csv"

    write_claimclaim_file(claim_claim, str(output_filename), docclaim_filetext)


    ##
    # in condition 6, write another claim/claim file for claims
    # that have a topic ID in common even if the query text is not the same
    if args.condition == "condition6":
        claim_claim = claimpairs_that_have_a_query_in_common(query_rel, query_filetext, docclaim_filetext, generalize_query_ids = True)

        output_filename = output_path / "claim_claim_for_relatedness.csv"

        write_claimclaim_file(claim_claim, str(output_filename), docclaim_filetext)

    ##
    # also write out query_rel, since for condition 6
    # this is the only record we have of which claims are both candidates and related
    output_filename = output_path / "query_related_claims.csv"

    # write csv output
    with open(str(output_filename), 'w', newline='') as csvfile:
        fieldnames = ["Query_ID", "Claim_ID"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for query_id, relatedclaims in query_rel.items():
            # in condition 6, our query_id is actually a pair (query_id, query_text).
            # for this file, only retain the actual query ID
            if args.condition == "condition6":
                if len(query_id) != 2:
                    print("Error, expected internal ID to be (ID, text) in condition 6, but I got:", query_id)
                    sys.exit(1)
                actual_query_id, query_text = query_id
                query_id = actual_query_id

            for claim_id in relatedclaims:
                writer.writerow({"Query_ID" : query_id, "Claim_ID" : claim_id} )
    

if __name__ == '__main__':
    main()
