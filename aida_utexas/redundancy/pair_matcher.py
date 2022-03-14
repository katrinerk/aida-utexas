import os
import csv
import ast
import argparse

''' read orginal queries.tsv with topic info to dictionary
	with format query_id: [query_filename, query_id, query...]'''
def queries_to_dict(filepath):
	queries = {}
	with open(filepath) as f:
		query_reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
		for line in query_reader:
			line[-1] = ast.literal_eval(line[-1])
			queries[line[1]] = line
	return queries


''' read orginal docclaims.tsv with topic info to dictionary
	with format claim_id: [claim_filename, claim_id, claim...]'''
def docclaims_to_dict(filepath):
	docclaims = {}
	with open(filepath) as f:
		claim_reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
		for line in claim_reader:
			line[-1] = ast.literal_eval(line[-1])
			docclaims[line[1]] = line
	return docclaims


def check_match(queries, docclaims, query_id, claim_id):
	# check topic match
	if queries[query_id][-3] == docclaims[claim_id][-3]:
		# check subtopic match
		if queries[query_id][-2] == docclaims[claim_id][-2]:
			#check template match
			for template in docclaims[claim_id][-1]:
				if template in queries[query_id][-1]:
					return True
	return False


def find_matching(queries, docclaims, redundency_output):
	matching = []
	nli = []

	for line in redundency_output:
		# select claims above a given threshold
		if line[-2] == "Related":
			query_id = line[1]
			claim_id = line[4]
			# select claims with matching topic, subtopic and template
			if check_match(queries, docclaims, query_id, claim_id):
				query_filename = line[0]
				claim_filename = line[3]
				matching.append([query_filename, query_id, claim_filename, claim_id])
				nli.append(line)

	return matching, nli


"""write matching output to csv"""
def get_matching(query_file, docclaim_file, redundancy_file, output_path):
	
	queries = queries_to_dict(query_file)
	docclaims = docclaims_to_dict(docclaim_file)

	# read redundency classifier output
	with open(redundancy_file) as f:
		redundancy_reader = csv.reader(f)
		redundancy_output = [line for line in redundancy_reader]

	matching, _ = find_matching(queries, docclaims, redundancy_output)

	fields = ['Query_Filename', 'Query_ID', 'Claim_Filename', 'Claim_ID'] 

	with open(output_path, 'w', newline='') as csvfile: 
	    csvwriter = csv.writer(csvfile) 
	    csvwriter.writerow(fields) 
	    csvwriter.writerows(matching)


"""write nli input to csv"""
def get_nli_input(query_file, docclaim_file, redundancy_file, output_path):

	queries = queries_to_dict(query_file)
	docclaims = docclaims_to_dict(docclaim_file)

	# read redundency classifier output
	with open(redundancy_file) as f:
		redundancy_reader = csv.reader(f)
		redundancy_output = [line for line in redundancy_reader]

	_, nli_input = find_matching(queries, docclaims, redundancy_output)

	fields = ['Query_Filename', 'Query_ID', 'Query_Sentence', 'Claim_Filename', 'Claim_ID', 'Claim_Sentence', 'Redundant_or_Independent', 'Score'] 

	with open(output_path, 'w',  newline='') as csvfile: 
	    csvwriter = csv.writer(csvfile) 
	    csvwriter.writerow(fields) 
	    csvwriter.writerows(nli_input)



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, required=True, help="for example: ta2_colorado")

	parser.add_argument('--condition', type=str, required=True, help="condition5, contition6, condition7")

	parser.add_argument('--docclaim_file', type=str, required=False, default="../../evaluation_2022/dryrun_data/preprocessed", 
							help="path to preprocessed docclaims")

	parser.add_argument('--query_file', type=str, required=False, default="../../evaluation_2022/dryrun_data/preprocessed/query", 
							help="path to preprocessed query")

	parser.add_argument('--redundancy_file', type=str, required=False, default="../../evaluation_2022/dryrun_data/working", 
							help="path to redundancy classifier output file")

	parser.add_argument('--output_path', type=str, required=False, default = "../../evaluation_2022/dryrun_data/working", 
							help="path to working space for output result")

	parser.add_argument('--write_matching', type=str, required=False, default = "True", 
							help="write matched query_claim pairs")

	parser.add_argument('--write_nli_input', type=str, required=False, default = "True", 
							help="write matched query_claim pairs in the format consuming by NLI model")
	args = parser.parse_args()

	docclaim_file = os.path.join(args.docclaim_file, args.data, "docclaims.tsv")
	query_file = os.path.join(args.query_file, args.condition, "queries.tsv")
	redundancy_file = os.path.join(args.redundancy_file, args.data, args.condition, "step1_query_claim_relatedness/q2d_relatedness.csv")

	if args.write_matching == "True":
		output_path = os.path.join(args.output_path, args.data, args.condition, "step3_claim_claim_ranking/query_claim_matching.csv")
		get_matching(query_file, docclaim_file, redundancy_file, output_path)

	if args.write_nli_input == "True":
		output_path = os.path.join(args.output_path, args.data, args.condition, "step2_query_claim_nli/nli_input.csv")
		get_nli_input(query_file, docclaim_file, redundancy_file, output_path)



if __name__ == '__main__':
    main()