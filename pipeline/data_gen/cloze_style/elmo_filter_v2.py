import numpy as np
import io
import os
import re
import json
import dill
import torch
import h5py
import re
from collections import defaultdict
from copy import deepcopy
from adapted_data_to_input_match_subtask import *
from adapted_data_to_input import Indexer
from elmo_tokenize import get_rej_ere_ids
import bcolz
import gensim

mix_list_train = os.listdir('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New/Train')
mix_list_val = os.listdir('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New/Val')
mix_list_test = os.listdir('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New/Test')

train_graph_list = set()
val_graph_list = set()
test_graph_list = set()

bad_files_train = []
bad_files_val = []
bad_files_test = []

for list_iter, list_item in enumerate([mix_list_train, mix_list_val, mix_list_test]):
    for file_iter, file_name in enumerate(list_item):
        if 'self' in file_name:
            if list_iter == 0:
                graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt = dill.load(open(os.path.join('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New/Train', file_name), 'rb'))
                train_graph_list.add(graph.eres[ent_ere_id].graph_id)
            elif list_iter == 1:
                graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt = dill.load(open(os.path.join('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New/Val', file_name), 'rb'))
                val_graph_list.add(graph.eres[ent_ere_id].graph_id)
            elif list_iter == 2:
                graph, query_stmts, res_stmts, ent_ere_id, target_event_stmt = dill.load(open(os.path.join('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New/Test', file_name), 'rb'))
                test_graph_list.add(graph.eres[ent_ere_id].graph_id)

            cand_stmt = [stmt_id for stmt_id in graph.eres[ent_ere_id].stmt_ids if stmt_id in res_stmts][0]
            graph_ids = [graph.graph_id]
        else:
            if list_iter == 0:
                graph, query_stmts, source_ere_id, ent_ere_ids, target_event_stmt = dill.load(open(os.path.join('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New/Train', file_name), 'rb'))
            elif list_iter == 1:
                graph, query_stmts, source_ere_id, ent_ere_ids, target_event_stmt = dill.load(open(os.path.join('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New/Val', file_name), 'rb'))
            elif list_iter == 2:
                graph, query_stmts, source_ere_id, ent_ere_ids, target_event_stmt = dill.load(open(os.path.join('/home/atomko/backup_drive/Summer_2020/Debug_Mixtures_New/Test', file_name), 'rb'))

            ent_ere_id = graph.stmts[target_event_stmt].tail_id

            graph_ids = list({stmt_id[:26] for stmt_id in graph.eres[ent_ere_id].stmt_ids})

            if list_iter == 0:
                train_graph_list.update(graph_ids)
            elif list_iter == 1:
                val_graph_list.update(graph_ids)
            elif list_iter == 2:
                test_graph_list.update(graph_ids)

            target_graph_id = graph.stmts[list(query_stmts)[0]].graph_id

            trg_ent_ere_id = list(set(ent_ere_ids) - {ent_ere_id})[0]

            assert len(graph_ids) == 2

        for stmt_id, stmt in graph.stmts.items():
            head_id = stmt.head_id
            tail_id = stmt.tail_id

            if not tail_id:
                assert stmt_id in graph.eres[head_id].stmt_ids
            else:
                assert stmt_id in graph.eres[head_id].stmt_ids
                assert stmt_id in graph.eres[tail_id].stmt_ids
                assert tail_id in graph.eres[head_id].neighbor_ere_ids
                assert head_id in graph.eres[tail_id].neighbor_ere_ids

        for ere_id, ere in graph.eres.items():
            assert ere_id not in ere.neighbor_ere_ids
            assert len([item for item in ere.stmt_ids if not graph.stmts[item].tail_id]) == 1
            assert graph.stmts[[item for item in ere.stmt_ids if not graph.stmts[item].tail_id][0]].graph_id == ere.graph_id

            assert set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} if graph.stmts[stmt_id].tail_id else {graph.stmts[stmt_id].head_id} for stmt_id in ere.stmt_ids]) - {ere_id} == ere.neighbor_ere_ids

            temp = set()

            for stmt_id, stmt in graph.stmts.items():
                if ere_id in [stmt.head_id, stmt.tail_id]:
                    temp.add(stmt_id)

            assert temp == ere.stmt_ids

        # if 'self' in file_name:
        #     if len({stmt_id for stmt_id in graph.stmts.keys() if stmt_id not in res_stmts and graph.stmts[stmt_id].tail_id}) >= 5:
        #         num_hops_query = random.randint(4, 4)
        #
        #         seen_eres = set()
        #         curr_eres = {ent_ere_id}
        #
        #         curr_num_hops = 0
        #
        #         stmt_id_sets = []
        #
        #         while curr_num_hops < num_hops_query and len(curr_eres) > 0:
        #             seen_eres.update(curr_eres)
        #
        #             stmt_ids = set.union(*[set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id]) for ere_id in curr_eres]) - {target_event_stmt}
        #
        #             if curr_num_hops > 0:
        #                 stmt_ids -= set.union(*stmt_id_sets[:curr_num_hops])
        #
        #             #rel_stmts_to_add = set()
        #
        #             for stmt_id in stmt_ids:
        #                 # if graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
        #                 #     rel_stmts_to_add.update([item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id])
        #                 #     curr_eres.update(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids)
        #
        #                 curr_eres.update([graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id])
        #
        #             #stmt_ids.update(rel_stmts_to_add)
        #
        #             if stmt_ids:
        #                 stmt_id_sets.append(stmt_ids)
        #
        #             curr_eres -= seen_eres
        #             curr_num_hops += 1
        #
        #         if len(set.union(*stmt_id_sets)) < 5:
        #             assert len({item for item in query_stmts if graph.stmts[item].tail_id}) == len(set.union(*stmt_id_sets))
        #         else:
        #             assert len({item for item in query_stmts if graph.stmts[item].tail_id}) >= 5
        #     else:
        #         # print(file_name, query_stmts)
        #         # # print({item for item in query_stmts if graph.stmts[item].tail_id})
        #         # # print({stmt_id for stmt_id in graph.stmts.keys() if stmt_id not in res_stmts and graph.stmts[stmt_id].tail_id})
        #         # if file_name != 'enwiki-20100312_0000729750_http:_www.isi.edu_gaia_entities_59c84e8c-2377-484b-9783-e860219651ec-self.p':
        #         try:
        #             assert len({item for item in query_stmts if graph.stmts[item].tail_id}) == len({stmt_id for stmt_id in graph.stmts.keys() if stmt_id not in res_stmts and graph.stmts[stmt_id].tail_id})
        #         except AssertionError:
        #             if list_iter == 0:
        #                 bad_files_train.append(file_name)
        #             elif list_iter == 1:
        #                 bad_files_val.append(file_name)
        #             elif list_iter == 2:
        #                 bad_files_test.append(file_name)
        # else:
        #     if len({stmt_id for stmt_id in graph.stmts.keys() if graph.stmts[stmt_id].graph_id == graph.stmts[list(query_stmts)[0]].graph_id and graph.stmts[stmt_id].tail_id}) >= 5:
        #         num_hops_query = random.randint(4, 4)
        #
        #         seen_eres = set()
        #         curr_eres = {ent_ere_id}
        #
        #         curr_num_hops = 0
        #
        #         stmt_id_sets = []
        #
        #         while curr_num_hops < num_hops_query and len(curr_eres) > 0:
        #             seen_eres.update(curr_eres)
        #
        #             stmt_ids = set.union(*[set([stmt_id for stmt_id in graph.eres[ere_id].stmt_ids if graph.stmts[stmt_id].tail_id]) for ere_id in curr_eres]) - {target_event_stmt}
        #
        #             if curr_num_hops > 0:
        #                 stmt_ids -= set.union(*stmt_id_sets[:curr_num_hops])
        #
        #             #rel_stmts_to_add = set()
        #
        #             for stmt_id in stmt_ids:
        #                 # if graph.eres[graph.stmts[stmt_id].head_id].category == 'Relation':
        #                 #     rel_stmts_to_add.update(
        #                 #         [item for item in graph.eres[graph.stmts[stmt_id].head_id].stmt_ids if graph.stmts[item].tail_id])
        #                 #     curr_eres.update(graph.eres[graph.stmts[stmt_id].head_id].neighbor_ere_ids)
        #
        #                 curr_eres.update([graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id])
        #
        #             #stmt_ids.update(rel_stmts_to_add)
        #
        #             if stmt_ids:
        #                 stmt_id_sets.append(stmt_ids)
        #
        #             curr_eres -= seen_eres
        #             curr_num_hops += 1
        #
        #         if len(set.union(*stmt_id_sets)) < 5:
        #             assert len({item for item in query_stmts if graph.stmts[item].tail_id}) == len(set.union(*stmt_id_sets))
        #         else:
        #             assert len({item for item in query_stmts if graph.stmts[item].tail_id}) >= 5
        #     else:
        #         assert len({item for item in query_stmts if graph.stmts[item].tail_id}) == len({stmt_id for stmt_id in graph.stmts.keys() if graph.stmts[stmt_id].graph_id == graph.stmts[list(query_stmts)[0]].graph_id and graph.stmts[stmt_id].tail_id})

        if 'self' not in file_name:
            # assert len([item for item in graph.eres[ent_ere_id].stmt_ids if graph.stmts[item].graph_id != graph.eres[ent_ere_id].graph_id]) == 1
            assert target_event_stmt not in query_stmts

            for item in query_stmts:
                assert graph.stmts[item].graph_id != graph.stmts[target_event_stmt].graph_id

            # for stmt_id, stmt in graph.stmts.items():
            #     if stmt_id != target_event_stmt and stmt.tail_id:
            #         assert graph.eres[stmt.head_id].graph_id == graph.eres[stmt.tail_id].graph_id
        # else:
        #     assert not set.intersection(query_stmts, res_stmts)
        #
        #     res_eres = set.union(*[{graph.stmts[res_stmt].head_id, graph.stmts[res_stmt].tail_id} if graph.stmts[res_stmt].tail_id else {graph.stmts[res_stmt].head_id} for res_stmt in res_stmts if res_stmt != target_event_stmt])
        #
        #     for stmt_id, stmt in graph.stmts.items():
        #         if stmt_id != target_event_stmt and stmt.tail_id:
        #             stmt = graph.stmts[stmt_id]
        #             print(graph.eres[stmt.head_id].label, stmt.label, graph.eres[stmt.tail_id].label)
        #             print(stmt_id)
        #             for item in set.intersection({stmt.head_id, stmt.tail_id}, res_eres):
        #                 print(graph.eres[item].label)
        #             assert len(set.intersection({stmt.head_id, stmt.tail_id}, res_eres)) in [0, 2]

        query_eres = set.union(*[{graph.stmts[stmt_id].head_id, graph.stmts[stmt_id].tail_id} if graph.stmts[stmt_id].tail_id else {graph.stmts[stmt_id].head_id} for stmt_id in query_stmts])
        assert len(query_eres) == len([item for item in query_stmts if not graph.stmts[item].tail_id])

        for ere_id in graph.eres.keys():
            assert len({item for item in graph.eres[ere_id].stmt_ids if not graph.stmts[item].tail_id}) == 1

        print(file_iter)

    assert not set.intersection(train_graph_list, val_graph_list)
    assert not set.intersection(train_graph_list, test_graph_list)
    assert not set.intersection(val_graph_list, test_graph_list)

    if list_iter == 0:
        dill.dump(bad_files_train, open('/home/atomko/backup_drive/Summer_2020/' + 'other_bad_files_v2_train.p', 'wb'))
    elif list_iter == 1:
        dill.dump(bad_files_val, open('/home/atomko/backup_drive/Summer_2020/' + 'other_bad_files_v2_val.p', 'wb'))
    elif list_iter == 2:
        dill.dump(bad_files_test, open('/home/atomko/backup_drive/Summer_2020/' + 'other_bad_files_v2_test.p', 'wb'))