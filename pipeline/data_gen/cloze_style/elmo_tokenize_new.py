import numpy as np
import io
import os
import re
import json
import dill
import torch
import h5py
from collections import defaultdict
from copy import deepcopy
from sacremoses import MosesTokenizer, MosesDetokenizer

graph_list = os.listdir('/home/atomko/backup_drive/Summer_2020/Graph_Singles')

curr_graphs = set([item.split('.')[0] + '.p' for item in os.listdir('/home/atomko/H5_Samples') if '.h5' in item])

print(len(graph_list))

graph_list = list(set(graph_list) - curr_graphs)

print(len(graph_list))

from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder(weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
                    options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json',
                    cuda_device=0)

tokenizer = MosesTokenizer()
detokenizer = MosesDetokenizer()

redo_files = []

for file_iter, file_name in enumerate(graph_list):
    try:
        graph = dill.load(open(os.path.join('/home/atomko/backup_drive/Summer_2020/Graph_Singles', file_name), 'rb'))
        #print(graph)
        if len(graph.eres.keys()) == 0 or (len(graph.stmts.keys()) == 0):
            continue

        in_file = open(os.path.join('/home/atomko/WikiRPI-text', graph.graph_id + '.rsd.txt'), 'r')
        input_str = in_file.read().rstrip() + '\n'
        in_file.close()

        input_str = re.sub('\xa0', ' ', input_str)
        input_str = re.sub('\u3000', ' ', input_str)
        input_str = re.sub('\u2009', ' ', input_str)
        input_str = re.sub('\u2028', ' ', input_str)
        input_str = re.sub('\u200a', ' ', input_str)

        in_file = open(os.path.join('/home/atomko/WikiRPI-text', graph.graph_id + '.rsd.txt'), 'r')
        lines = in_file.readlines()

        input_lines = [re.sub('\xa0', ' ', x.strip()) for x in lines]
        input_lines = [re.sub('\u3000', ' ', x.strip()) for x in input_lines]
        input_lines = [re.sub('\u2009', ' ', x.strip()) for x in input_lines]
        input_lines = [re.sub('\u2028', ' ', x.strip()) for x in input_lines]
        input_lines = [re.sub('\u200a', ' ', x.strip()) for x in input_lines]

        in_file.close()

        newline_ind = []

        for iter, item in enumerate(input_str):
            if item == '\n':
                newline_ind.append(iter)

        line_des = dict()

        line_begins = [0] + [item + 1 for item in newline_ind[:-1]]

        for iter in range(len(newline_ind)):
            if iter == 0:
                line_des[iter] = (0, newline_ind[iter])
            else:
                line_des[iter] = ((newline_ind[iter - 1] + 1), newline_ind[iter])

            iter += 1

        jj_file = open(os.path.join('/home/atomko/jj-wiki/', graph.graph_id + '.jsonjust'))
        jj_obj = json.load(jj_file)
        jj_file.close()

        js_file = open(os.path.join('/home/atomko/js-wiki-new-ontology/', graph.graph_id + '.json'))
        js_obj = json.load(js_file)['theGraph']
        js_file.close()

        just_ind = defaultdict(set)

        for key, value in jj_obj.items():
            for item in value:
                start = int(item['startOffset'][0])
                end = int(item['endOffsetInclusive'][0]) + 1
                just_ind[graph.graph_id + '_' + key].add((start, end))

        ere_ids_to_type_stmts = dict()

        for key, value in js_obj.items():
            if value['type'] == 'Statement':
                if value['predicate'] == 'type':
                    ere_id = value['subject']
                    ere_ids_to_type_stmts[graph.graph_id + '_' + ere_id] = graph.graph_id + '_' + key

        ere_dict = defaultdict(lambda: defaultdict(dict))

        for ere_id, ere in graph.eres.items():
            for stmt_id in ere.stmt_ids:
                ere_dict[ere_id][stmt_id] = just_ind[stmt_id]

        line_cov = defaultdict(set)

        for ere_id, ere in graph.eres.items():
            list_ind = [line_des[key] for key in sorted(list(line_des.keys()))]

            for ind in set([item for sublist in list(ere_dict[ere_id].values()) for item in sublist]):
                start = None
                end = None

                for iter in range(len(list_ind)):
                    if ind[0] >= list_ind[iter][0] and ind[0] < list_ind[iter][1]:
                        start = iter

                    if ind[1] > list_ind[iter][0] and ind[1] <= list_ind[iter][1]:
                        end = iter

                    if start and end:
                        break

                line_cov[ere_id].add((start, end))

        for ere_id in line_cov.keys():
            for item in line_cov[ere_id]:
                assert item[0] == item[1], 'Not a single line'

        list_tok = []

        all_line_ind = sorted(list(set.union(*[set.union(*[{item[0]} for item in line_cov[ere_id]]) for ere_id in line_cov.keys()])))

        rev_map = {item : ind for ind, item in enumerate(all_line_ind)}

        narrow_temp = [item for iter, item in enumerate(input_lines) if iter in all_line_ind]

        for item in narrow_temp:
            tokens = tokenizer.tokenize(item, escape=False)
            list_tok.append(tokens)

        embs = elmo.embed_sentences(list_tok)

        char_to_tok_map = dict()

        text_list = []

        for iter, tokens in enumerate(list_tok):
            char_to_tok_map[all_line_ind[iter]] = dict()

            line_begin = line_begins[all_line_ind[iter]]

            orig_sen = input_lines[all_line_ind[iter]]

            char_count = line_begins[all_line_ind[iter]]

            for sub_iter in range(len(list_tok[iter])):
                text = detokenizer.detokenize(list_tok[iter][:sub_iter + 1])

                char_mark = 0

                indices_to_del = []

                indices_to_insert = []

                char_iter = 0

                while char_iter < len(text):
                    item = text[char_iter]

                    if item != orig_sen[char_mark] and item == ' ':
                        indices_to_del.append(char_iter)
                        char_iter += 1
                    elif item != orig_sen[char_mark] and orig_sen[char_mark] == ' ':
                        indices_to_insert.append(char_iter)
                        char_mark += 1
                    else:
                        char_mark += 1
                        char_iter += 1

                text_map = {ind_iter : str(item) for ind_iter, item in enumerate(text)}

                for ind_to_insert in indices_to_insert:
                    text_map[ind_to_insert] = ' ' + text_map[ind_to_insert]

                text = ''.join([text_map[other_iter] for other_iter in sorted(list(text_map.keys())) if other_iter not in indices_to_del])

                for char_ind in range(char_count, line_begin + len(text)):
                    char_to_tok_map[all_line_ind[iter]][char_ind] = sub_iter

                char_count = len(text) + line_begin

            text_list.append(text)

        check = True

        try:
            for iter, line in enumerate(text_list):
                temp = input_str[min(list(char_to_tok_map[all_line_ind[iter]].keys())):max(list(char_to_tok_map[all_line_ind[iter]].keys())) + 1]

                assert temp == input_lines[all_line_ind[iter]]
                assert temp == line
        except AssertionError:
            check = False

        if not check:
            redo_files.append(file_name)
            continue

        line_cov_stmts = dict()

        for stmt_id in [stmt_id for stmt_id in graph.stmts.keys() if graph.stmts[stmt_id].tail_id]:
            list_ind = [line_des[key] for key in sorted(list(line_des.keys()))]

            assert len(just_ind[stmt_id]) == 1

            ind = list(just_ind[stmt_id])[0]
            start = None
            end = None

            for iter in range(len(list_ind)):
                if ind[0] >= list_ind[iter][0] and ind[0] < list_ind[iter][1]:
                    start = iter

                if ind[1] > list_ind[iter][0] and ind[1] <= list_ind[iter][1]:
                    end = iter

                if start and end:
                    break

            line_cov_stmts[stmt_id] = (start, end)

        tok_map = dict()

        for stmt_id in line_cov_stmts.keys():
            line_ind = line_cov_stmts[stmt_id][0]
            ind = list(just_ind[stmt_id])[0]

            tok_ind = (char_to_tok_map[line_ind][ind[0]], char_to_tok_map[line_ind][ind[1] - 1])

            tok_map[stmt_id] = (line_ind, tok_ind)

        elmo_embs = list(embs)

        list_stmt_embs = []
        list_stmts = []

        for stmt_id in tok_map.keys():
            emb_line = rev_map[tok_map[stmt_id][0]]
            tok_inds = tok_map[stmt_id][1]

            stmt_embs = np.expand_dims(np.average(elmo_embs[emb_line][:, tok_inds[0]:tok_inds[1] + 1, :], axis=1), axis=0)
            list_stmts.append(stmt_id)
            list_stmt_embs.append(stmt_embs)

        h5_file = h5py.File('/home/atomko/H5_Samples/' + graph.graph_id + '.h5')

        emb_mat = np.concatenate(list_stmt_embs, axis=0)

        non_type_group = h5_file.create_group('non_type_group')

        dataset = non_type_group.create_dataset('Non-typing Statement Embeddings', data=emb_mat, dtype=np.float32)

        dill.dump(list_stmts, open('/home/atomko/H5_Samples/non_type_stmt_names_' + graph.graph_id + '.p', 'wb'))

        line_cov_type_stmts = defaultdict(set)

        for stmt_id in [stmt_id for stmt_id in graph.stmts.keys() if not graph.stmts[stmt_id].tail_id]:
            list_ind = [line_des[key] for key in sorted(list(line_des.keys()))]

            inds = just_ind[stmt_id]

            for ind in inds:
                start = None
                end = None

                for iter in range(len(list_ind)):
                    if ind[0] >= list_ind[iter][0] and ind[0] < list_ind[iter][1]:
                        start = iter

                    if ind[1] > list_ind[iter][0] and ind[1] <= list_ind[iter][1]:
                        end = iter

                    if start and end:
                        break

                line_cov_type_stmts[stmt_id].add(((start, end), ind))

        type_tok_map = defaultdict(set)

        for stmt_id in line_cov_type_stmts.keys():
            for item in line_cov_type_stmts[stmt_id]:
                line_ind = item[0][0]
                ind = item[1]

                tok_ind = (char_to_tok_map[line_ind][ind[0]], char_to_tok_map[line_ind][ind[1] - 1])

                type_tok_map[stmt_id].add((line_ind, tok_ind))

        type_stmt_emb_map = dict()

        arg_map = defaultdict(lambda: defaultdict(dict))

        master_other_emb_list = []
        other_stmt_list = []

        for stmt_id in type_tok_map.keys():
            ere_id = graph.graph_id + '_' + js_obj['http' + stmt_id.split('_http')[1]]['subject']

            if graph.eres[ere_id].category in ['Event', 'Relation']:
                arg_stmts = [item for item in graph.eres[ere_id].stmt_ids if graph.stmts[item].tail_id and graph.stmts[item].head_id == ere_id]
            elif graph.eres[ere_id].category == 'Entity':
                arg_stmts = [item for item in graph.eres[ere_id].stmt_ids if graph.stmts[item].tail_id and graph.stmts[item].tail_id == ere_id]

            arg_lines = [tok_map[arg_stmt][0] for arg_stmt in arg_stmts]

            line_map = {item : iter for iter, item in enumerate(sorted(list(set(arg_lines))))}

            rev_line_map = {iter : item for (item, iter) in line_map.items()}

            for num in range(len(line_map.keys())):
                arg_map[stmt_id][num] = set([item for arg_iter, item in enumerate(arg_stmts) if line_map[arg_lines[arg_iter]] == num])

            assert len(set.intersection(set(line_map.keys()), set([item[0] for item in type_tok_map[stmt_id]]))) == len(set(line_map.keys()))

            type_stmt_embs = []

            ind_sets = dict()

            for line_num in line_map.keys():
                ind_sets[line_num] = set([item for item in type_tok_map[stmt_id] if item[0] == line_num])

            for iter, item in enumerate([ind_sets[line_num] for line_num in sorted(list(ind_sets.keys()))]):
                master_emb_list = []

                for sub_item in item:
                    emb_line = rev_map[sub_item[0]]
                    tok_inds = sub_item[1]
                    #print(emb_line, len(elmo_embs))
                    stmt_emb = np.expand_dims(np.average(elmo_embs[emb_line][:, tok_inds[0]:tok_inds[1] + 1, :], axis=1), axis=0)

                    master_emb_list.append(stmt_emb)

                if len(master_emb_list) == 1:
                    type_stmt_embs.append(master_emb_list[0])
                else:
                    all_embs = np.concatenate(master_emb_list, axis=0)
                    type_stmt_embs.append(np.expand_dims(np.average(all_embs, axis=0), axis=0))

            type_stmt_emb_map[stmt_id] = np.concatenate(type_stmt_embs, axis=0)

            other_emb_list = []

            other_inds = set([item[0] for item in type_tok_map[stmt_id]]) - set(line_map.keys())

            for item in [sub_item for sub_item in type_tok_map[stmt_id] if sub_item[0] in other_inds]:
                emb_line = rev_map[item[0]]
                tok_inds = item[1]

                stmt_emb = np.expand_dims(np.average(elmo_embs[emb_line][:, tok_inds[0]:tok_inds[1] + 1, :], axis=1), axis=0)

                other_emb_list.append(stmt_emb)

            if len(other_inds) > 0:
                other_emb_mat = np.expand_dims(np.average(np.concatenate(other_emb_list, axis=0), axis=0), axis=0)
                master_other_emb_list.append(other_emb_mat)
                other_stmt_list.append(stmt_id)

            if graph.eres[ere_id].category in ['Event', 'Relation']:
                assert set.union(*list(arg_map[stmt_id].values())) == set([item for item in graph.eres[ere_id].stmt_ids if graph.stmts[item].tail_id and graph.stmts[item].head_id == ere_id]), 'Nope'
            elif graph.eres[ere_id].category == 'Entity':
                assert set.union(*list(arg_map[stmt_id].values())) == set([item for item in graph.eres[ere_id].stmt_ids if graph.stmts[item].tail_id and graph.stmts[item].tail_id == ere_id]), 'Nope'

            for key in arg_map[stmt_id].keys():
                for item in arg_map[stmt_id][key]:
                    assert tok_map[item][0] == rev_line_map[key]

        type_group = h5_file.create_group('type_group')

        other_emb = dict()

        for iter in range(len(master_other_emb_list)):
            other_emb[other_stmt_list[iter]] = master_other_emb_list[iter]

        for item in type_stmt_emb_map.keys():
            new_dataset = type_group.create_dataset(item.split('_http')[0] + '_http:__www.isi.edu_gaia_assertions_' + item.split('/')[-1], data=type_stmt_emb_map[item], dtype=np.float32)

            if item in other_stmt_list:
                new_dataset.attrs['other_emb'] = other_emb[item]

        h5_file.close()

        dill.dump(arg_map, open('/home/atomko/H5_Samples/type_to_non_map_' + graph.graph_id + '.p', 'wb'))

        print(file_iter, graph.graph_id)
    except RuntimeError as e:
        redo_files.append(file_name)
        print(e)

dill.dump(redo_files, open('/home/atomko/redo_files.p', 'wb'))