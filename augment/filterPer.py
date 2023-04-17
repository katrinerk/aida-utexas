import csv
from wikipedia2vec import Wikipedia2Vec


wiki2vec = Wikipedia2Vec.load("/home/cc/wikipedia2vec/enwiki_20180420_300d.pkl")


with open('people_full.csv', mode='r') as csv_file:
    with open('people_filtered.csv', 'w') as csv_output_file:
        csv_reader = csv.DictReader(csv_file,fieldnames=['name'])
        csv_writer = csv.writer(csv_output_file)
        csv_writer.writerow(['name'])
        for row in csv_reader:
            print(row['name'])
            try:                
                emb = wiki2vec.get_entity_vector(row['name'])
                print("Get ", row['name'])
            except:
                print("--------no this entity", row['name'])
            else:
                csv_writer.writerow([row['name']])