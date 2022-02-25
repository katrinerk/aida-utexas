import os
import sys

# find relative path_jy
from pathlib import Path
currpath = Path(os.getcwd())

sys.path.insert(0, str(currpath.parents[1]))

import io
import json
import shutil
import logging
from argparse import ArgumentParser
from collections import defaultdict

from aida_utexas.aif.aida_graph import AidaGraph
from aida_utexas.aif.json_graph import JsonGraph
from aida_utexas import util


# parse files containing claim frames and possibly other EREs and statements
# write as json files,
# extract information about each claim and return as dictionary
def parse_aifdocs(inpath, output_dir, fileinfix = ""):
    in_filepaths = util.get_file_list(inpath, suffix=".ttl")

    claims = [ ]
    
    for inpath in in_filepaths:

        # transform and write query file
        aif_graph = AidaGraph()
        aif_graph.build_graph(str(inpath), fmt = "ttl")
        json_graph = JsonGraph()
        json_graph.build_graph(aif_graph)

        output_path = output_dir / (inpath.name + fileinfix + '.json')
        fout = open(output_path, "w")
        json.dump(json_graph.as_dict(), fout, indent=1)
        fout.close()

        # extract claim texts and claim IDs
        for claim_label, claim_entry in json_graph.each_claim():
            if claim_entry.text is not None:
                thisclaim = {
                    "file" : inpath.name,
                    "claim_id" : json_graph.shorten_label(claim_label),
                    "claim_text" : claim_entry.text,
                    "topic" : claim_entry.topic,
                    "subtopic" : claim_entry.subtopic,
                    "claim_template" : str(claim_entry.claim_template)
                    }
                claims.append(thisclaim)

    return claims

def write_claiminfo_to_file(claims, fout):
    for c in claims:
        print(c["file"], c["claim_id"], c["claim_text"], c["topic"], c["subtopic"], c["claim_template"], sep = "\t", file = fout)


def main():
    ######3
    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument('aif_path', help='Path to document-level AIF files')
    parser.add_argument('doc_output_dir', help='Directory to write document-level output. tsv will be here, json files in a subdirectory called json')
    parser.add_argument('-q', '--query_path',
                        help='Path to the directory with conditions 5,6,7 query info (or empty)', default = None, type = str)
    parser.add_argument('-Q', '--query_output_dir', help='Directory to write query files. Will have Condition 5,6,7 subdirectories with tsv files, json files in Condition 5/json', default = None, type = str)
    parser.add_argument('-f', '--force', action='store_true',
                        help='If specified, overwrite existing output files without warning')
    
    
    args = parser.parse_args()
    
    ####
    
    
    ####
    # making query jsons, retaining texts and IDs
    if args.query_path is not None:

        # check for existence, make sure we have a pathlib Path
        query_in_path = util.get_input_path(args.query_path)

        # make output directory.
        # this will overwrite everything in this output directory! 
        query_out_dir = util.get_output_path(args.query_output_dir, overwrite_warning=not args.force)

        for condition in ["Condition5", "Condition6", "Condition7"]:
            # determine input directory
            this_in_dir = query_in_path / condition

            # determine output directory
            this_out_dir = util.get_output_dir(query_out_dir / condition,  overwrite_warning=not args.force)

            # copy topics tsv
            topicfilename = this_in_dir / "topics.tsv"
            shutil.copy(str(topicfilename), this_out_dir)

            
            # in condition 5, convert query ttl to json and make queries.tsv
            if condition == "Condition5":
                
                query_out_json_dir = util.get_output_dir(this_out_dir / "json/", overwrite_warning=not args.force)
                query_in_ttl_dir = util.get_input_path(this_in_dir / "Query_Claim_Frames")
        
                claims = parse_aifdocs(query_in_ttl_dir, query_out_json_dir, "_query")
        
                output_path = this_out_dir / "queries.tsv"
                fout = open(output_path, "w")
                
                write_claiminfo_to_file(claims, fout)
                
                fout.close()

    #####
    # making document jsons, retaining texts and IDS
    doc_out_dir = util.get_output_dir(args.doc_output_dir, overwrite_warning=not args.force)
    doc_out_json_dir = util.get_output_dir(doc_out_dir / "json/", overwrite_warning=not args.force)

    output_path = doc_out_dir / "docclaims.tsv"
    fout = open(output_path, "w")

    # possibly run through multiple subdirectories
    for root, dirs, _ in os.walk(args.aif_path):
        claims = parse_aifdocs(root, doc_out_json_dir)
        write_claiminfo_to_file(claims, fout)
        # for thisdir in dirs:
        #     fulldir = os.path.join(root, thisdir)
        #     claims = parse_aifdocs(fulldir, doc_out_json_dir)
        #     write_claiminfo_to_file(claims, fout)
            
    fout.close() 
        

if __name__ == '__main__':
    main()