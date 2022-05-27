import sys
import os
import logging
import csv
import requests
import json
from pathlib import Path
currpath = Path(os.getcwd())

sys.path.insert(0, str(currpath.parents[1]))

def main():
    
        
    search_api = "https://kgtk.isi.edu/api"
    similarity_api = "https://kgtk.isi.edu/similarity_api"
    object1 = "70% of the German population"
    object2 = "alkaline diets"
    
    global q1, q2
    
    response = requests.get(search_api + "?q=" + object1 + "&extra_info=true&language=en&is_class=false&type=ngram")
    result = response.json()
    print(json.dumps(result, indent = 4))
    cnt = 1

    for obj in result:
        if cnt == 1:
            q1 = (obj["qnode"])
            print (q1)
            break

    
    response2 = requests.get(search_api + "?q=" + object2 + "&extra_info=true&language=en&is_class=false&type=ngram")
    result2 = response2.json()
    for obj in result:
        if cnt == 2:
            q2 = (obj["qnode"])
            print (q2)
            break
        cnt = cnt +1
    print(json.dumps(result2, indent = 4))

    
            
    
    # q1 = "Q148"
    # q2 = "NILQ00006686"
    #https://kgtk.isi.edu/similarity_api?q1=Q144&q2=Q146&similarity_type=class
    #topsim, class, jc, complex, transe, text
    similarity_set = {"topsim", "class", "jc", "complex", "transe", "text"}
    for similarity_type in similarity_set:
        query_string = similarity_api + "?q1=" + q1 + "&q2=" + q2 + "&similarity_type=" + similarity_type
        response3 = requests.get(query_string)
        print(response3.status_code)
        print("similarity type: " + similarity_type)
        result3 = response3.json()
        print(json.dumps(result3, indent = 4))
    
    
if __name__ == '__main__':
    main()