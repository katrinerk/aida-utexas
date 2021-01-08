# KE dec 2020
# adapted from examine_clusterseeds
#
# This examines the output of a run of our system on a given SIN
# to answer the question: What nodes were found to
# match the entry points in that SIN, and what weight
# were the matches assigned?
#
# A great match is one with a weight > 90.
# Ideally we would like to see, for each entry point, a single
# matching node with a weight > 90.
#
# There is reason for concern if, for a given entry point,
# - we only see matching nodes with weights < 90, or 
# - we see more than 2 matching nodes with a weight > 90
# 
#
# usage:
# python3 analyze_entrypoints.py <query jsons directory>
# 
# This is a directory found at <outputdir>/WORKING/query_jsons/
#
# The script lists first, for each entry point, the number of good matches
# (good = weight >90),
# then it lists the node IDs and weights for each match.
#
# This script can be used to analyze the quality of entry points,
# or, in preparation for analyze_hypotheses.py,
# as a way to identify entry points for further manual analysis. 

import sys
import json
from pathlib import Path
from typing import List, Union

def get_input_path(path: Union[str, Path], check_exist: bool = True) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if check_exist:
        assert path.exists(), f'{path} does not exist!'

    return path


seeds_dir = Path(sys.argv[1]).resolve()
assert seeds_dir.exists() and seeds_dir.is_dir(), \
    '{} does not exist!'.format(seeds_dir)

# list of query_jsons files
seeds_file_list = sorted(
    [f for f in seeds_dir.iterdir() if f.suffix == '.json'])

for seeds_file in seeds_file_list:
    file_path = get_input_path(seeds_file, check_exist=True)
    with open(file_path, "r") as fin:
        seeds_json = json.load(fin)

    sin_name = seeds_file.stem.split("_")[0]

    # overall assessment of goodness
    print("====SIN:", sin_name, "=======")

    for entrypoint in seeds_json["ep_matches_dict"].keys():
        good_eps = [e for e in seeds_json["ep_matches_dict"][entrypoint] if e[1] > 90]
        print("EP:", entrypoint, "#good:", len(good_eps))



for seeds_file in seeds_file_list:
    file_path = get_input_path(seeds_file, check_exist=True)
    with open(file_path, "r") as fin:
        seeds_json = json.load(fin)

    sin_name = seeds_file.stem.split("_")[0]
    
    # top entry points
    print("====SIN:", sin_name, "=======")

    for entrypoint in seeds_json["ep_matches_dict"].keys():
        for ename, eprob in seeds_json["ep_matches_dict"][entrypoint][:4]:
            print(entrypoint, eprob, ":", ename)
