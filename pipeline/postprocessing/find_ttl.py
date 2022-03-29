import os, fnmatch
import pandas
from pathlib import Path

def find(pattern, path):
    result = {}
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                #result.append(os.path.join(root, name))
                currpath = os.path.join(root, name)
                dir = os.path.basename(os.path.dirname(currpath))
                result[dir] = currpath
    return result

def relevant_file_filter(csvfile_path):
    filepath = Path(csvfile_path)       
    record = pandas.read_csv(filepath,sep='\t', names=["location", "claim_id", "topic", "subtopic", "template", "detail"])
    
    unique_claim = record['claim_id'].unique()
    claim_ttl = {}
    for row in record.itertuples(index=True, name='Pandas'):
        location = str(row.location).split("/") 
        claim_ttl[str(row.claim_id)] = location[0]
            
    return claim_ttl

def main():
    result = relevant_file_filter("/Users/cookie/Downloads/docclaims.tsv")
    for claim in result.keys():
        print(claim + ":" + result[claim])
    
if __name__ == '__main__':
    main()