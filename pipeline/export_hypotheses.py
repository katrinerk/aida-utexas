from collections import defaultdict
from argparse import ArgumentParser
import re
import sys
import copy
from aida_utexas import util
from aida_utexas.soin import SOIN
from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis.aida_hypothesis import AidaHypothesisCollection


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_path', help='path to the graph JSON file')
    parser.add_argument('hypothesis_path', help='path to the JSON file with hypotheses')
    parser.add_argument('roles_ontology_path', help='path to the roles ontology file')
    parser.add_argument('output_dir', help='directory to write human-readable hypotheses')

    args = parser.parse_args()

    json_graph = JsonGraph.from_dict(util.read_json_file(args.graph_path, 'JSON graph'))

    hypotheses_json = util.read_json_file(args.hypothesis_path, 'hypotheses')
    hypothesis_collection = AidaHypothesisCollection.from_json(hypotheses_json, json_graph)

    roles_ontology = util.read_json_file(args.roles_ontology_path, 'roles ontology')

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=True)

    output_list = []
    for hypo_idx, hypothesis in enumerate(hypothesis_collection.hypotheses):
        output_path = output_dir / 'hypothesis-{:0>3d}.txt'.format(hypo_idx)
        result = hypothesis.to_str_for_csv(roles_ontology)
        with open(str(output_path), "w", encoding="utf-8") as fout:
            print(result, file=fout)
        result = result.replace(',', ' &').replace('ID: ', '')
        result_list = result.replace('\n    ', ',').split('\n\n')
        for ere_idx, res in enumerate(result_list):
            tmp_res_list = res.split(',')
            if res:
                if len(tmp_res_list[1]) < 2 or tmp_res_list[1][:2] not in 'T1T2T3T4':
                    tmp_res_list.insert(1, '')
                for _ in range(9 - len(tmp_res_list)):
                    tmp_res_list.insert(-1, '')
                for idx, tmp_res in enumerate(tmp_res_list):
                    if len(tmp_res.split(': ')) == 2 and tmp_res.split(': ')[1] == '':
                        tmp_res_list[idx] = ''
                    
                for question_ID in hypothesis.questionIDs:
                    question_ID = '_'.join(question_ID.split('_')[3:]) 
                    sin_info = question_ID + '.{}.{}'.format(hypo_idx + 1, ere_idx + 1)
                    sin_info_list = sin_info.replace('.', '_').split('_')
                    sin_info_list = tuple([int(''.join([i for i in x if i.isdigit()])) for x in sin_info_list])
                    tmp_res_list2 = copy.deepcopy(tmp_res_list)
                    tmp_res_list2.insert(0, sin_info)
                    res = ','.join(tmp_res_list2)
                    output_list.append((sin_info_list, res))
 
    output_list.sort()
    csv_output_path = output_dir / args.hypothesis_path.split('/')[-1].replace('json', 'csv')
    with open(csv_output_path, 'w', encoding="utf-8") as csv_file:
        csv_file.write('SIN,Event or Relation type,time,arg1,arg2,arg3,arg4,arg5,comments,ID\n')   
        prev = tuple()
        for idx, output in enumerate(output_list):
            if idx != 0 and prev[0] != output[0][0]:
                csv_file.write('\n')
            if idx != 0 and prev[1] != output[0][1]:
                csv_file.write('\n')
            if idx != 0 and prev[2] != output[0][2]:
                csv_file.write('\n')
            if idx != 0 and prev[3] != output[0][3]:
                csv_file.write('\n')
            csv_file.write(output[1] + '\n')
            prev = output[0]

if __name__ == '__main__':
    main()