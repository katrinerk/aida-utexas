"""Sentence Transformers is a library that is like Huggingface's Transformers, 
but pools the final hidden state output such that cosine similarity can be measured between two sequences. 
The models here are further trained from their Huggingface equivalents on a dataset of 
[over 1 billion pairs](https://huggingface.co/datasets/sentence-transformers/embedding-training-data) 
trained to detect similarity."""

#python redundancy_score.py --data ta2_colorado --type claim_claim

import os
import argparse
import csv
from collections import defaultdict
import copy
from sentence_transformers import SentenceTransformer, util

# read the docclaims.tsvs
def read_doc(doc_input):

	claims_split_by_document = defaultdict(list)
	claims_to_id = dict()

	with open(doc_input) as file:

	    tsv_file = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
	    for line in tsv_file:
	        claims_to_id[line[2]] = (line[0], line[1])
	        claims_split_by_document[0].append(line[2])

	return claims_split_by_document, claims_to_id


# read the queries.tsvs
def read_query(query_input):
	queries_split_by_document = defaultdict(list)
	queries_to_id = dict()

	with open(query_input) as file:

	    tsv_file = csv.reader(file, delimiter="\t",quoting=csv.QUOTE_NONE)
	    for line in tsv_file:
	        queries_to_id[line[2]] = (line[0], line[1])
	        queries_split_by_document[0].append(line[2])
	return queries_split_by_document, queries_to_id


# read claim_claim.csv for claims pairwise redundancy score calculation
def read_combined(input_file):
	queries_split_by_document = defaultdict(list)
	queries_to_id = dict()

	claims_split_by_document = defaultdict(list)
	claims_to_id = dict()

	with open(input_file) as file:
		reader = csv.reader(file)
		next(reader, None)
		for line in reader:
			queries_to_id[line[2]] = (line[0], line[1])
			queries_split_by_document[0].append(line[2])
			
			claims_to_id[line[5]] = (line[3], line[4])
			claims_split_by_document[0].append(line[5])

	return queries_split_by_document, queries_to_id, claims_split_by_document, claims_to_id


"""write output to csv"""
def write_output(output_path, rows, header):
	# field names 
	if header == "query_claim":
		fields = ['Query_Filename', 'Query_ID', 'Query_Sentence', 'Claim_Filename', 'Claim_ID', 'Claim_Sentence', 'Redundant_or_Independent', 'Score'] 
	elif header == "claim_claim":
		fields = ['Claim1_Filename', 'Claim1_ID', 'Claim1_Sentence', 'Claim2_Filename', 'Claim2_ID', 'Claim2_Sentence', 'Redundant_or_Independent', 'Score'] 

	with open(output_path, 'w') as csvfile: 

	    csvwriter = csv.writer(csvfile) 
	    # writing the fields 
	    csvwriter.writerow(fields) 
	    # writing the data rows
	    csvwriter.writerows(rows)


"""Calculate redundancy/independent of query_claim pair 

This is a sample model I chose - it's 22M parameters and a distilled version of BERT.
By default, the model uses a GPU if it is available.
"""
def calculate_redundancy_query_claim(claims_split_by_document, claims_to_id, queries_split_by_document, queries_to_id, threshold):

	model = SentenceTransformer('all-MiniLM-L6-v2')
	rows = []
	queries = queries_split_by_document[0]

	sentences = copy.deepcopy(claims_split_by_document[0])
	sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

	query_num = -1
	for query in queries:

		query_num += 1
		query_embeddings = model.encode([query], convert_to_tensor=True)
		#Compute cosine-similarities
		cosine_scores = util.cos_sim(query_embeddings, sentence_embeddings)

		for i in range(len(cosine_scores[0])):
			score = cosine_scores[0][i].item()
			label = 'Related' if score >= threshold else 'Unrelated'
			
			score = round(score, 2)

			query_filename, query_claim_id = queries_to_id[query]
			doc_filename, doc_claim_id = claims_to_id[sentences[i]]
			rows.append([query_filename, query_claim_id, query, doc_filename, doc_claim_id, sentences[i], label, score]) 
	return rows


"""Calculate redundancy/independent of query_claim pair 

This is a sample model I chose - it's 22M parameters and a distilled version of BERT.
By default, the model uses a GPU if it is available.
"""
def calculate_redundancy_claim_claim(claims_split_by_document, claims_to_id, queries_split_by_document, queries_to_id, threshold):

	model = SentenceTransformer('all-MiniLM-L6-v2')
	rows = []
	queries = queries_split_by_document[0]
	claims = claims_split_by_document[0]

	query_num = -1
	for i, query in enumerate(queries):

		query_num += 1
		claim = claims[i]
		query_embeddings = model.encode([query], convert_to_tensor=True)
		claim_embeddings = model.encode([claim], convert_to_tensor=True)
		#Compute cosine-similarities
		cosine_scores = util.cos_sim(query_embeddings, claim_embeddings)

		score = cosine_scores[0][0].item()
		label = 'Related' if score >= threshold else 'Unrelated'
		
		score = round(score, 2)

		query_filename, query_claim_id = queries_to_id[query]
		doc_filename, doc_claim_id = claims_to_id[claim]
		rows.append([query_filename, query_claim_id, query, doc_filename, doc_claim_id, claim, label, score]) 
	return rows


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--type', type=str, required=True, help="query_claim or claim_claim")

	parser.add_argument('--condition', type=str, required=True, help="condition5, contition6, condition7")

	parser.add_argument('--threshold', type=float, required=False, default=0.58, 
						help="the threshold used to separate redundant from independent sentences.")

	parser.add_argument('--data', type=str, required=True, help="for example: ta2_colorado")

	parser.add_argument('--docclaim_file', type=str, required=False, default="../../evaluation_2022/dryrun_data/preprocessed", 
						help="path to preprocessed docclaims")

	parser.add_argument('--query_file', type=str, required=False, default="../../evaluation_2022/dryrun_data/preprocessed/query", 
						help="path to preprocessed query")

	parser.add_argument('--output_path', type=str, required=False, default = "../../evaluation_2022/dryrun_data/working", 
						help="path to working space for output result")
	args = parser.parse_args()

	if args.type == "query_claim":

		docclaim_file = os.path.join(args.docclaim_file, args.data, "docclaims.tsv")
		query_file = os.path.join(args.query_file, args.condition, "queries.tsv")
		output_path = os.path.join(args.output_path, args.data, args.condition, "step1_query_claim_relatedness/q2d_relatedness.csv")

		claims_split_by_document, claims_to_id = read_doc(docclaim_file)
		queries_split_by_document, queries_to_id = read_query(query_file)
		rows = calculate_redundancy_query_claim(claims_split_by_document, claims_to_id, queries_split_by_document, queries_to_id, args.threshold)
		write_output(output_path, rows, args.type)

	if args.type == "claim_claim":
		input_file = os.path.join(args.output_path, args.data, args.condition, "step3_claim_claim_ranking/claim_claim.csv")
		output_path = os.path.join(args.output_path, args.data, args.condition, "step3_claim_claim_ranking/claim_claim_redundancy.csv")

		# claim1 is as query, claim2 is as claim
		queries_split_by_document, queries_to_id, claims_split_by_document, claims_to_id = read_combined(input_file)
		rows = calculate_redundancy_claim_claim(claims_split_by_document, claims_to_id, queries_split_by_document, queries_to_id, args.threshold)
		write_output(output_path, rows, args.type)

if __name__ == '__main__':
    main()
