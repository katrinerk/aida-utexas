"""
gen_plausibility_samples.py
~~~~~~~~~~~~~~~~~~~~~~~~
Author: Yejin Cho; 2020.

(1) Positive samples
- Merge point events with all of its arguments that come from the same source graph

(2) Negative samples
- Either a merge point event with *all* its adjacent edges or some subset of these edges that come from different graphs

Usage:
    $ python3 gen_plausibility_samples.py <input-graph-dir> <output-dir>
    (e.g., python3 gen_plausibility_samples.py 5k_Graph_Salads ps_5k_Graph_Salads)
    
"""
import dill
import random
import os
from tqdm import tqdm
from collections import defaultdict
import argparse

def gen_neg_sample(core_graph_ids, graph_id_to_stmt_map, used_clusters):
    dup = True

    while dup:
        stmt_ids = set()

        for graph_id in core_graph_ids:
            stmt_pool = graph_id_to_stmt_map[graph_id]
            rand_int = random.randint(1, len(stmt_pool))

            stmt_ids = set.union(stmt_ids, set(random.sample(list(stmt_pool), rand_int)))

        if stmt_ids not in used_clusters:
            dup = False

    return stmt_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_neg_smp_per_merge_pt", type=int, default=3)

    args = parser.parse_args()
    locals().update(vars(args))

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_groups = ['Train', 'Val', 'Test']

    [os.makedirs(os.path.join(output_dir, data_group)) for data_group in data_groups if not os.path.exists(os.path.join(output_dir, data_group))]

    for data_group in data_groups:
        input_path = os.path.join(input_dir, data_group)

        # List all graph salads in the input_path
        salad_fname_list = os.listdir(input_path)

        # Loop over all graph salads
        for salad_fname in tqdm(salad_fname_list):
            # Load graph salad
            graph_dict = dill.load(open(os.path.join(input_path, salad_fname), 'rb'))
            graph_mix = graph_dict['graph_mix']
            ere_mat_ind = graph_dict['ere_mat_ind']
            query_ere_ids = {ere_mat_ind.get_word(ere_ind) for ere_ind in graph_dict['query_eres']}
            origin_id = [ere_id for ere_id in query_ere_ids if graph_dict['graph_mix'].eres[ere_id].category == 'Event'][0]

            # Graph ids (of subgraphs)
            core_graph_ids = [graph_dict['target_graph_id']] + list({graph_mix.stmts[stmt_id].graph_id for stmt_id in graph_mix.eres[origin_id].stmt_ids} - {graph_dict['target_graph_id']})

            # Find merge points
            merge_ere_ids = set()

            for ere_id, ere in graph_mix.eres.items():
                if len(set.intersection({graph_mix.stmts[stmt_id].graph_id for stmt_id in ere.stmt_ids}, set(core_graph_ids))) == len(core_graph_ids):
                    merge_ere_ids.add(ere_id)

            assert not set.intersection(merge_ere_ids, graph_dict['noisy_merge_points'])

            # Generate positive and negative samples for plausibility classifier for hypothesis seeds
            pos_samples = defaultdict(list)
            neg_samples = defaultdict(list)

            for merge_ere_id in merge_ere_ids:  # loop over merge points
                graph_id_to_stmt_map = dict()

                for core_graph_id in core_graph_ids:
                    graph_id_to_stmt_map[core_graph_id] = {stmt_id for stmt_id in graph_mix.eres[merge_ere_id].stmt_ids if graph_mix.stmts[stmt_id].tail_id and graph_mix.stmts[stmt_id].graph_id == core_graph_id}

                # Generate three positive samples
                # Do not use merge point with only one argument from a source
                pos_samples[merge_ere_id] = [value for value in graph_id_to_stmt_map.values() if len(value) > 1]

                used_clusters = set()

                # Generate negative samples using the existing neighbors from different sources
                for _ in range(num_neg_smp_per_merge_pt):
                    neg_cluster = gen_neg_sample(core_graph_ids, graph_id_to_stmt_map, used_clusters)

                    neg_samples[merge_ere_id].append(neg_cluster)
                    used_clusters.add(frozenset(neg_cluster))

                for sample_iter in range(len(pos_samples[merge_ere_id])):
                    ere_ids = set.union(*[{graph_mix.stmts[stmt_id].head_id, graph_mix.stmts[stmt_id].tail_id} for stmt_id in pos_samples[merge_ere_id][sample_iter]])

                    for ere_id in ere_ids:
                        pos_samples[merge_ere_id][sample_iter].update({stmt_id for stmt_id in graph_mix.eres[ere_id].stmt_ids if not graph_mix.stmts[stmt_id].tail_id})

                for sample_iter in range(len(neg_samples[merge_ere_id])):
                    ere_ids = set.union(*[{graph_mix.stmts[stmt_id].head_id, graph_mix.stmts[stmt_id].tail_id} for stmt_id in neg_samples[merge_ere_id][sample_iter]])

                    for ere_id in ere_ids:
                        neg_samples[merge_ere_id][sample_iter].update({stmt_id for stmt_id in graph_mix.eres[ere_id].stmt_ids if not graph_mix.stmts[stmt_id].tail_id})

            # Save pickled graph
            dill.dump({'pos_samples': pos_samples, 'neg_samples': neg_samples}, open(os.path.join(output_dir, data_group, salad_fname), "wb"))


# # --------------------------------------------------------------------------------------------------------------
# # NB. Verification process
# # --------------------------------------------------------------------------------------------------------------
# # For each merge point (e.g., five)
# for m in range(len(merge_ere_ids)):
#     merge_m = merge_ere_ids[m]
#     pos_samples_m = pos_samples[merge_m]
#     # For each of three positive samples
#     for v in range(len(pos_samples_m)):
#         print('------------------------------')
#         print('Sample # {}'.format(v))
#         print('------------------------------')
#         # For each element statement in the positive sample
#         for w in range(len(pos_samples_m[v])):
#             sample = pos_samples_m[v][w]
#             src_graph = graph_ids_dict[graph_mix.stmts[sample].graph_id]
#             print("Graph # {}".format(src_graph))
#
#
# # For each merge point (e.g., five)
# for m in range(len(merge_ere_ids)):
#     merge_m = merge_ere_ids[m]
#     neg_samples_m = neg_samples[merge_m]
#     # For each of three negative samples
#     for v in range(len(neg_samples_m)):
#         print('------------------------------')
#         print('Sample # {}'.format(v))
#         print('------------------------------')
#         # For each element statement in the negative sample
#         for w in range(len(neg_samples_m[v])):
#             sample = neg_samples_m[v][w]
#             src_graph = graph_ids_dict[graph_mix.stmts[sample].graph_id]
#             print("Graph # {}".format(src_graph))
#
