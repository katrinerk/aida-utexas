import os
import csv
import ast
import argparse

from pathlib import Path

########
# given a queries.tsv or docclaims.tsv file, return:
# mapping from query_id to [filename, id, text, topic, subtopic, template]
def query_or_claim_to_dict(filepath):
	mapping = {}
	with open(filepath) as f:
		reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
		for line in reader:
			line[-1] = ast.literal_eval(line[-1])
			mapping[line[1]] = line
	return mapping



########
# check if the pair (query, docclaim) have matched topic, 
# subtopic and template
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



##########
# given filepath to queries.tsv & docclaims.tsv & redundancy score:
# return all docclaims that are RELATED to query 
#							and have topic, subtopic and template matched
def find_matching(queries, docclaims, redundancy_output):
	matching = []
	nli = []

	# column indexes in redundancy output
	# ['Query_Filename', 'Query_ID', 'Query_Sentence', 'Claim_Filename', 
	# 'Claim_ID', 'Claim_Sentence', 'Redundant_or_Independent', 'Score']

	for line in redundancy_output:
		# select all related docclaims
		if line[-2] == "Related":
			query_id = line[1]
			claim_id = line[4]
			# select docclaims with matched topic, subtopic and template
			if check_match(queries, docclaims, query_id, claim_id):
				query_filename = line[0]
				claim_filename = line[3]
				# nli keep more fields for data analysis
				matching.append([query_filename, query_id, claim_filename, claim_id])
				nli.append(line)

	return matching, nli


###########
# wrap up all matching steps
# write matched docclaims to csv for later NLI task
def get_nli_input(query_file, docclaim_file, redundancy_file, output_path):

	docclaims = query_or_claim_to_dict(docclaim_file)
	queries = query_or_claim_to_dict(query_file)

	# read redundancy classifier output
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
	parser.add_argument('--run', type=str, required=True, help="for example: ta2_colorado")

	parser.add_argument('--condition', type=str, required=True, 
							help="condition5, contition6, condition7")

	parser.add_argument('--docclaim_dir', type=str, required=True, 
							help="path to preprocessed docclaims")
	
	parser.add_argument('--query_dir', type=str, required=True, 
							help="path to preprocessed query")

	parser.add_argument('--workspace', type=str, required=True, 
							help="path the directory for processing work")

	args = parser.parse_args()

	docclaim_file = Path(os.path.join(args.docclaim_dir, args.run, "docclaims.tsv"))
	query_file = Path(os.path.join(args.query_dir, args.condition, "queries.tsv"))
	redundancy_file = Path(os.path.join(args.workspace, args.run, args.condition, "step1_query_claim_relatedness/q2d_relatedness.csv"))

	output_dir = os.path.join(args.workspace, args.run, "condition5/step2_query_claim_nli")
	if not Path(output_dir).exists():
		os.makedirs(output_dir)
	output_path = Path(os.path.join(output_dir, "nli_input.csv"))
	get_nli_input(query_file, docclaim_file, redundancy_file, output_path)

if __name__ == '__main__':
    main()