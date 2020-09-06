import numpy as np
import io
import os
import re

import json
import dill
from collections import defaultdict
from copy import deepcopy

def check_poss_ere_stmt_spans(graph, ere_id, res_stmts):
    focus_span = get_stmt_spans(graph)

    res_side_spans = set.union(*[focus_span[res_stmt_id] for res_stmt_id in {item for item in res_stmts if graph.stmts[item].tail_id}])

    query_stmts_to_keep = set()

    for stmt_id in {item for item in graph.eres[ere_id].stmt_ids if graph.stmts[item].tail_id}:
        if not set.intersection(focus_span[stmt_id], res_side_spans):
            query_stmts_to_keep.add(stmt_id)

    return query_stmts_to_keep

def check_poss_stmt_span(graph, poss_res_stmts, res_stmts, rej_eres, add_other_event_stmts, add_other_rel_stmts):
    rej_stmts = set()

    if rej_eres:
        rej_stmts = set.union(*[{item for item in graph.eres[rej_ere].stmt_ids if graph.stmts[item].tail_id} for rej_ere in rej_eres])

    query_side_stmts = set.union(({item for item in graph.stmts.keys() if graph.stmts[item].tail_id} - set.union(res_stmts, poss_res_stmts, rej_stmts)), set.union(add_other_event_stmts, add_other_rel_stmts))

    focus_span = get_stmt_spans(graph)

    poss_res_stmts_to_add = set()

    query_side_spans = set.union(*[focus_span[other_stmt_id] for other_stmt_id in query_side_stmts])

    for stmt_id in poss_res_stmts:
        if not set.intersection(query_side_spans, focus_span[stmt_id]):
            poss_res_stmts_to_add.add(stmt_id)

    return poss_res_stmts_to_add

def check_spans_distinct(graph, temp, res_stmts):
    focus_span = get_stmt_spans(graph)

    res_stmt_spans = set.union(*[focus_span[stmt_id] for stmt_id in res_stmts if graph.stmts[stmt_id].tail_id])
    other_stmt_spans = set.union(*[focus_span[stmt_id] for stmt_id in graph.stmts.keys() if graph.stmts[stmt_id].tail_id and stmt_id not in set.union(temp, res_stmts)])

    for key in {item for item in focus_span.keys() if item in res_stmts and graph.stmts[item].tail_id}:
        span = focus_span[key]
        for other_key in focus_span.keys():
            if other_key in graph.stmts.keys() and other_key not in temp:
                if other_key not in res_stmts and graph.stmts[other_key].tail_id:
                    if set.intersection(span, focus_span[other_key]):
                        print('What', key, other_key)

    return not set.intersection(res_stmt_spans, other_stmt_spans)

def get_rej_ere_ids(graph, res_stmts):
    focus_span = get_stmt_spans(graph)

    ere_dict = defaultdict(set)

    for ere_id, ere in graph.eres.items():
        if ere.category in ['Event', 'Relation']:
            for stmt_id in ere.stmt_ids:
                if graph.stmts[stmt_id].tail_id:
                    ere_dict[ere_id].update(focus_span[stmt_id])

    ere_id_list = set()

    res_side_spans = set.union(*[focus_span[res_stmt_id] for res_stmt_id in {item for item in res_stmts if graph.stmts[item].tail_id}])

    for key in ere_dict.keys():
        ind = ere_dict[key]

        if not set.intersection({item for item in graph.eres[key].stmt_ids if graph.stmts[item].tail_id}, res_stmts):
            if set.intersection(ind, res_side_spans):
                ere_id_list.add(key)

    return ere_id_list

def check_distinct_ere_pair(graph, ere_id_1, ere_id_2):
    in_file = open(os.path.join('/home/cc/WikiRPI-text', graph.graph_id + '.rsd.txt'), 'r')
    input_str = in_file.read().rstrip() + '\n'
    in_file.close()

    newline_ind = []

    for iter, item in enumerate(input_str):
        if item == '\n':
            newline_ind.append(iter)

    line_des = dict()

    for iter in range(len(newline_ind)):
        if iter == 0:
            line_des[iter] = (0, newline_ind[iter])
        else:
            line_des[iter] = ((newline_ind[iter - 1] + 1), newline_ind[iter])

        iter += 1

    jj_file = open(os.path.join('/home/cc/jj-wiki/', graph.graph_id + '.jsonjust'))
    jj_obj = json.load(jj_file)
    jj_file.close()

    just_ind = defaultdict(set)

    for stmt_id in [item for item in graph.eres[ere_id_1].stmt_ids if graph.stmts[item].tail_id]:
        for item in jj_obj[('').join(stmt_id.split('_')[2:])]:
            start = int(item['startOffset'][0])
            end = int(item['endOffsetInclusive'][0]) + 1
            just_ind[stmt_id].add((start, end))

    for stmt_id in [item for item in graph.eres[ere_id_2].stmt_ids if graph.stmts[item].tail_id]:
        for item in jj_obj[('').join(stmt_id.split('_')[2:])]:
            start = int(item['startOffset'][0])
            end = int(item['endOffsetInclusive'][0]) + 1
            just_ind[stmt_id].add((start, end))

    ere_dict = defaultdict(lambda: defaultdict(dict))

    ere_1 = graph.eres[ere_id_1]

    for stmt_id in ere_1.stmt_ids:
        if graph.stmts[stmt_id].tail_id:
            ere_dict[ere_id_1][stmt_id] = just_ind[stmt_id]

    ere_2 = graph.eres[ere_id_2]

    for stmt_id in ere_2.stmt_ids:
        if graph.stmts[stmt_id].tail_id:
            ere_dict[ere_id_2][stmt_id] = just_ind[stmt_id]

    line_cov = defaultdict(set)

    list_ind = [line_des[key] for key in sorted(list(line_des.keys()))]

    for ere_id in [ere_id_1, ere_id_2]:
        for ind in set.union(*ere_dict[ere_id].values()):
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

    focus_span = defaultdict(set)

    for ere_id in [ere_id_1, ere_id_2]:
        for line_span in line_cov[ere_id]:
            char_span = (line_des[line_span[0]][0], line_des[line_span[1]][1])

            span_min = np.inf
            span_max = -1 * np.inf

            for ind in set([item for sublist in list(ere_dict[ere_id].values()) for item in sublist]):
                if ind[0] >= char_span[0] and ind[1] <= char_span[1]:
                    if ind[0] < span_min:
                        span_min = ind[0]

                    if ind[1] > span_max:
                        span_max = ind[1]

            focus_span[ere_id].add(char_span)#(span_min, span_max))

    distinct = not set.intersection(focus_span[ere_id_1], focus_span[ere_id_2])

    return distinct

def get_stmt_spans(graph):
    in_file = open(os.path.join('/home/cc/WikiRPI-text', graph.graph_id + '.rsd.txt'), 'r')
    input_str = in_file.read().rstrip() + '\n'
    in_file.close()

    input_str = re.sub('\xa0', ' ', input_str)
    input_str = re.sub('\u3000', ' ', input_str)
    input_str = re.sub('\u2009', ' ', input_str)
    input_str = re.sub('\u2028', ' ', input_str)
    input_str = re.sub('\u200a', ' ', input_str)

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

    jj_file = open(os.path.join('/home/cc/jj-wiki/', graph.graph_id + '.jsonjust'))
    jj_obj = json.load(jj_file)
    jj_file.close()

    just_ind = defaultdict(set)

    for key, value in jj_obj.items():
        if graph.graph_id + '_' + key in graph.stmts.keys() and graph.stmts[graph.graph_id + '_' + key].tail_id:
            for item in value:
                start = int(item['startOffset'][0])
                end = int(item['endOffsetInclusive'][0]) + 1
                just_ind[graph.graph_id + '_' + key].add((start, end))

    line_cov = defaultdict(set)

    list_ind = [line_des[key] for key in sorted(list(line_des.keys()))]

    for stmt_id in just_ind.keys():
        for ind in just_ind[stmt_id]:
            start = None
            end = None

            for iter in range(len(list_ind)):
                if ind[0] >= list_ind[iter][0] and ind[0] < list_ind[iter][1]:
                    start = iter

                if ind[1] > list_ind[iter][0] and ind[1] <= list_ind[iter][1]:
                    end = iter

                if start and end:
                    break

            line_cov[stmt_id].add((start, end))

    focus_span = defaultdict(set)

    for stmt_id in line_cov.keys():
        for line_span in line_cov[stmt_id]:
            char_span = (line_des[line_span[0]][0], line_des[line_span[1]][1])

            span_min = np.inf
            span_max = -1 * np.inf

            for ind in just_ind[stmt_id]:
                if ind[0] >= char_span[0] and ind[1] <= char_span[1]:
                    if ind[0] < span_min:
                        span_min = ind[0]

                    if ind[1] > span_max:
                        span_max = ind[1]

            focus_span[stmt_id].add(char_span)#(span_min, span_max))

    return focus_span