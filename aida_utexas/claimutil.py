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
def read_querydoc_relatedness(filepath, qid_claimcandidates, qid_2_file, cid_2_file, threshold = None):
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


