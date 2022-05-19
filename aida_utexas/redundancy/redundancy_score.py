"""Sentence Transformers is a library that is like Huggingface's Transformers, 
but pools the final hidden state output such that cosine similarity can be measured between two sequences. 
The models here are further trained from their Huggingface equivalents on a dataset of 
[over 1 billion pairs](https://huggingface.co/datasets/sentence-transformers/embedding-training-data) 
trained to detect similarity."""

import os
import argparse
import csv
from collections import defaultdict
import copy
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

######
# given a queries.tsv or docclaims.tsv file, return:
# txt_to_id: mapping from text to filenames and ID
# agg_txt: aggregate of all text
def read_query_or_docclaim(input):

	split_by_document = defaultdict(list)
	txt_to_id = dict()
	
	with open(input) as file:
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
		tsv_file = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
		for line in tsv_file:
			txt_to_id[line[2]] = (line[0], line[1])
			split_by_document[0].append(line[2])

	return split_by_document, txt_to_id


######
# given a claim_claim.csv file, return:
# claim1_to_id/claim2_to_id: mapping from claim to filenames and ID
# claim1_split_by_document/claim2_split_by_document: aggregate of all claim texts
def read_claim_claim_pair(input_file):
	claim1_split_by_document = defaultdict(list)
	claim1_to_id = dict()

	claim2_split_by_document = defaultdict(list)
	claim2_to_id = dict()

	# column indices:
	# 0: claim1_filename
	# 1: claim1_id
	# 2: claim1_text
	# 3: claim2_filename
	# 4: claim2_id
	# 5: claim2_text

	with open(input_file) as file:
		reader = csv.reader(file)
		next(reader, None)
		for line in reader:
			claim1_to_id[line[2]] = (line[0], line[1])
			claim1_split_by_document[0].append(line[2])

			claim2_to_id[line[5]] = (line[3], line[4])
			claim2_split_by_document[0].append(line[5])

	return claim1_split_by_document, claim1_to_id, claim2_split_by_document, claim2_to_id



##########
# calculate redundant/independent score for query-claim pair
# distilled version of BERT with 22M parameters
def calculate_redundancy_query_claim(queries_split_by_document, queries_to_id, claims_split_by_document, claims_to_id, threshold):

	model = SentenceTransformer('all-MiniLM-L6-v2')
	rows = []
	queries = queries_split_by_document[0]

	claims = copy.deepcopy(claims_split_by_document[0])
	claims_embeddings = model.encode(claims, convert_to_tensor=True)

	query_num = -1
	for query in queries:

		query_num += 1
		query_embeddings = model.encode([query], convert_to_tensor=True)
		#Compute cosine-similarities
		cosine_scores = util.cos_sim(query_embeddings, claims_embeddings)

		for i in range(len(cosine_scores[0])):
			score = cosine_scores[0][i].item()
			label = 'Related' if score >= threshold else 'Unrelated'
			
			score = round(score, 2)

			query_filename, query_claim_id = queries_to_id[query]
			claim_filename, claim_claim_id = claims_to_id[claims[i]]
			rows.append([query_filename, query_claim_id, query, claim_filename, claim_claim_id, claims[i], label, score]) 
	return rows


##########
# calculate redundant/independent score for claim-claim pair
def calculate_redundancy_claim_claim(claim1_split_by_document, claim1_to_id, claim2_split_by_document, claim2_to_id, threshold):

	model = SentenceTransformer('all-MiniLM-L6-v2')
	rows = []
	claim1_all = claim1_split_by_document[0]
	claim2_all = claim2_split_by_document[0]

	# calculate cosine-similarity between each claim-claim pair
	claim_num = -1
	for i, claim1 in enumerate(claim1_all):

		claim_num += 1
		claim2 = claim2_all[i]
		claim1_embeddings = model.encode([claim1], convert_to_tensor=True)
		claim2_embeddings = model.encode([claim2], convert_to_tensor=True)
		#Compute cosine-similarities
		cosine_scores = util.cos_sim(claim1_embeddings, claim2_embeddings)

		score = cosine_scores[0][0].item()
		label = 'Related' if score >= threshold else 'Unrelated'
		
		score = round(score, 2)

		claim1_filename, claim1_id = claim1_to_id[claim1]
		claim2_filename, claim2_id = claim2_to_id[claim2]
		rows.append([claim1_filename, claim1_id, claim1, claim2_filename, claim2_id, claim2, label, score]) 
	return rows


######
# write the relatedness output to file
def write_output(output_path, rows, header):
	# column indices for query-claim relatedness output
	if header == "query_claim":
		fields = ['Query_Filename', 'Query_ID', 'Query_Sentence', 'Claim_Filename', 'Claim_ID', 'Claim_Sentence', 'Redundant_or_Independent', 'Score'] 
	# column indices for claim-claim relatedness output
	elif header == "claim_claim":
		fields = ['Claim1_Filename', 'Claim1_ID', 'Claim1_Sentence', 'Claim2_Filename', 'Claim2_ID', 'Claim2_Sentence', 'Redundant_or_Independent', 'Score'] 

	with open(output_path, 'w') as csvfile: 
	    csvwriter = csv.writer(csvfile) 
	    # writing the fields 
	    csvwriter.writerow(fields) 
	    # writing the data rows
	    csvwriter.writerows(rows)
		

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--data', type=str, required=True, help="for example: ta2_colorado")

	parser.add_argument('--type', type=str, required=True, help="query_claim or claim_claim")

	parser.add_argument('--condition', type=str, required=True, help="condition5, contition6, condition7")

	parser.add_argument('--threshold', type=float, required=False, default=0.58, 
						help="the threshold used to separate redundant from independent sentences.")

	parser.add_argument('--docclaim_file', type=str, required=False, default="../../evaluation_2022/evaluation/preprocessed", 
						help="path to preprocessed docclaims")

	parser.add_argument('--query_file', type=str, required=False, default="../../evaluation_2022/evaluation/preprocessed/query", 
						help="path to preprocessed query")

	parser.add_argument('--output_path', type=str, required=False, default = "../../evaluation_2022/evaluation/working", 
						help="path to working space for output result")
	args = parser.parse_args()
	
	# sanity check on condition
	if args.condition not in ["condition5", "condition6", "condition7"]:
		print("Error: need a condition that is condition5, condition6, condition7")
		sys.exit(1)
	
	# sanity check on work type
	if args.type not in ["query_claim", "claim_claim"]:
		print("Error: need a type that is query_claim or claim_claim")
		sys.exit(1)

	if args.type == "query_claim":

		docclaim_file = Path(os.path.join(args.docclaim_file, args.data, args.condition, "docclaims.tsv"))
		query_file = Path(os.path.join(args.query_file, args.condition, "queries.tsv"))
		output_dir = os.path.join(args.output_path, args.data, args.condition, "step1_query_claim_relatedness")
		if not Path(output_dir).exists():
			os.mkdir(output_dir)
		output_path = Path(os.path.join(output_dir, "q2d_relatedness.csv"))

		claims_split_by_document, claims_to_id = read_query_or_docclaim(docclaim_file)
		queries_split_by_document, queries_to_id = read_query_or_docclaim(query_file)
		rows = calculate_redundancy_query_claim( queries_split_by_document, queries_to_id, claims_split_by_document, claims_to_id, args.threshold)
		write_output(output_path, rows, args.type)

	if args.type == "claim_claim":
		input_file = Path(os.path.join(args.output_path, args.data, args.condition, "step3_claim_claim_ranking/claim_claim.csv"))
		
		output_dir = os.path.join(args.output_path, args.data, args.condition, "step3_claim_claim_ranking")
		if not Path(output_dir).exists():
			os.mkdir(output_dir)
		output_path = Path(os.path.join(output_dir, "claim_claim_relatedness.csv"))

		claim1_split_by_document, claim1_to_id, claim2_split_by_document, claim2_to_id = read_claim_claim_pair(input_file)
		rows = calculate_redundancy_claim_claim(claim1_split_by_document, claim1_to_id, claim2_split_by_document, claim2_to_id, args.threshold)
		write_output(output_path, rows, args.type)

if __name__ == '__main__':
    main()
