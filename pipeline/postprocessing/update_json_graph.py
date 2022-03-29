import os
import json
import fileinput

def main():
    filename = '/Users/cookie/Downloads/task2_kb.ttl.json'
    with open(filename, "r") as fout2:
        json_data = json.load(fout2)
        for obj in json_data:
            if obj == 'theGraph':
                for item in json_data['theGraph']:
                    for key in json_data['theGraph'][item]:
                        if key == 'predicate' and json_data['theGraph'][item][key] == 'ill_one':
                            json_data['theGraph'][item][key] = 'A1_ppt__sick/ill_one'
            

    # dump json to another file
    with open("/Users/cookie/Downloads/task2_kb_patched_cond7.ttl.json", "w") as fout:
        fout.write(json.dumps(json_data, indent=2))
        
    

if __name__ == '__main__':
    main()