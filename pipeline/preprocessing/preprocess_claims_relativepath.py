import os
import sys

# find relative path_jy
from pathlib import Path
currpath = Path(os.getcwd())

sys.path.insert(0, str(currpath.parents[1]))

import io
import json
import logging
#import sys
#import os
from argparse import ArgumentParser
from collections import defaultdict

from aida_utexas.aif.aida_graph import AidaGraph
from aida_utexas.aif.json_graph import JsonGraph
from aida_utexas import util


# parse files containing claim frames and possibly other EREs and statements
# write as json files,
# extract NL text and ID of each claim
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
                claims.append( (inpath.name, json_graph.shorten_label(claim_label), claim_entry.text) )

    return claims


def main():
    ######3
    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument('aif_path', help='Path to document-level AIF files')
    parser.add_argument('doc_output_dir', help='Directory to write document-level json files')
    parser.add_argument('tab_output_dir', help='Directory to write tables with text claims and topics')
    parser.add_argument('-q', '--query_path',
                        help='Path to the directory with query claim frames (or empty)', default = None, type = str)
    parser.add_argument('-Q', '--query_output_dir', help='Directory to write query json files (or empty)', default = None, type = str)
    parser.add_argument('-f', '--force', action='store_true',
                        help='If specified, overwrite existing output files without warning')
    
    
    args = parser.parse_args()
    
    tab_output_dir = util.get_output_dir(args.tab_output_dir, overwrite_warning=not args.force)

    ####
    
    
    ####
    # making query jsons, retaining texts and IDs
    if args.query_path is not None:

        query_output_dir = util.get_output_dir(args.query_output_dir, overwrite_warning=not args.force)
        claims = parse_aifdocs(args.query_path, query_output_dir, "_query")
        
        output_path = tab_output_dir / "queries.tsv"
        fout = open(output_path, "w")
        for claimfile, claim_label, claim_text in claims:
            #print(claimfile, claim_label, claim_text)
            print(claimfile, claim_label, claim_text, sep = "\t", file = fout)
        fout.close()

    #####
    # making document jsons, retaining texts and IDS
    doc_output_dir = util.get_output_dir(args.doc_output_dir, overwrite_warning=not args.force)

    output_path = tab_output_dir / "docclaims.tsv"
    fout = open(output_path, "w")

    # possibly run through multiple subdirectories
    for root, dirs, _ in os.walk(args.aif_path):
        for thisdir in dirs:
            fulldir = os.path.join(root, thisdir)
            claims = parse_aifdocs(fulldir, doc_output_dir)
        
            for claimfile, claim_label, claim_text in claims:
                #print(claimfile, claim_label, claim_text)
                print(claimfile, claim_label, claim_text, sep = "\t", file = fout)
            
    fout.close() 
        

if __name__ == '__main__':
    main()
