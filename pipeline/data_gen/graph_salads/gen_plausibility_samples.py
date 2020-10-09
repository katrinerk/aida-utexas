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
from utils import *
from itertools import chain, combinations
import random as rd
import sys
import os
from tqdm import tqdm

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def gen_neg_samples(g0, g1, g2):
    g1g2 = list(powerset(g1 + g2))
    g2g0 = list(powerset(g2 + g0))
    g0g1 = list(powerset(g0 + g1))
    neg_samples = [g0 + list(i) for i in g1g2] + [g1 + list(i) for i in g2g0] + [g2 + list(i) for i in g0g1]
    return neg_samples

def gen_1_neg_sample(g0, g1, g2):
    while True:
        n_g0 = rd.randint(0, len(g0))
        n_g1 = rd.randint(0, len(g1))
        n_g2 = rd.randint(0, len(g2))
        if sum([n_g0, n_g1, n_g2]) != n_g0 and sum([n_g0, n_g1, n_g2]) != n_g1 and sum([n_g0, n_g1, n_g2]) != n_g2:
            break
    neg_sample = rd.sample(g0, n_g0) + rd.sample(g1, n_g1) + rd.sample(g2, n_g2)
    return neg_sample

def gen_neg_samples_faster(g0, g1, g2):
    neg_samples = []
    neg_samples.append(gen_1_neg_sample(g0, g1, g2))
    neg_samples.append(gen_1_neg_sample(g0, g1, g2))
    neg_samples.append(gen_1_neg_sample(g0, g1, g2))
    return neg_samples

def get_stmt_between_eres(graph_mix, ere1, ere2):
    stmts = list(set.intersection(graph_mix.eres[ere1].stmt_ids, graph_mix.eres[ere2].stmt_ids))
    return stmts[0]


# Filename
input_dir = sys.argv[1]
output_dir = sys.argv[2]

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all graph salads in the input_dir
salad_fname_list = os.listdir(input_dir)

# Loop over all graph salads
for salad_fname in tqdm(salad_fname_list):
    # salad_fname = input_path

    # Load graph salad
    # graph_dict = dill.load(open("5k_Graph_Salads/" + salad_fname, 'rb'))
    graph_dict = dill.load(open(os.path.join(input_dir, salad_fname), 'rb'))
    graph_mix = graph_dict['graph_mix']

    # Graph ids (of three subgraphs)
    graph_ids = list({graph_mix.stmts[stmt_id].graph_id for stmt_id in graph_mix.stmts.keys()})
    graph_ids_dict = dict(zip(graph_ids, [0, 1, 2]))

    # Find merge points
    merge_ere_ids = set()
    for ere_id, ere in graph_mix.eres.items():
        if len({graph_mix.stmts[stmt_id].graph_id for stmt_id in ere.stmt_ids}) == 3:
            merge_ere_ids.add(ere_id)
    merge_ere_ids = list(merge_ere_ids)

    # Generate positive and negative samples for plausibility classifier for hypothesis seeds
    pos_samples = dict()
    neg_samples = dict()

    for m in range(len(merge_ere_ids)):  # loop over merge points
        merge_m = merge_ere_ids[m]
        # print("\nMerge point: {} ({})".format(graph_mix.eres[merge_m].label[0], graph_mix.eres[merge_m].category))
        merge_m_neigh = graph_mix.eres[merge_m].neighbor_ere_ids

        # Initialize neighbor sets of each source graph
        neigh_g0 = set()
        neigh_g1 = set()
        neigh_g2 = set()

        # For each neighbor around a merge point
        for n in range(len(merge_m_neigh)):
            neighbor_n = list(merge_m_neigh)[n]
            src_graph = graph_ids_dict[graph_mix.eres[neighbor_n].graph_id]
            stmt_between = get_stmt_between_eres(graph_mix, merge_m, neighbor_n)
            src_graph_stmt = graph_ids_dict[graph_mix.stmts[stmt_between].graph_id]

            # if src_graph != src_graph_stmt:
            #     print('****** Mismatch ******')
            #     print('ERE source: {} (Neighbor ERE: {}, Merge ERE: {})'.format(src_graph, neighbor_n, merge_m))
            #     print('STMT source: {} (STMT: {})'.format(src_graph_stmt, stmt_between))
            #
            # print("Graph # {}".format(src_graph_stmt))
            # print("Statement: {}".format(graph_mix.stmts[stmt_between].raw_label))
            # print("Neighbor: {}".format(graph_mix.eres[neighbor_n].label[0]))

            # Classify the neighboring statement by source graph
            if src_graph_stmt == 0:
                neigh_g0.add(stmt_between)
            elif src_graph_stmt == 1:
                neigh_g1.add(stmt_between)
            elif src_graph_stmt == 2:
                neigh_g2.add(stmt_between)

        # Convert to lists for easier indexing
        neigh_g0 = list(neigh_g0)
        neigh_g1 = list(neigh_g1)
        neigh_g2 = list(neigh_g2)

        # Generate three positive samples
        # Do not use merge point with only one argument from a source
        pos_samples_list = []
        for neigh_gx in [neigh_g0, neigh_g1, neigh_g2]:
            if len(neigh_gx) > 1:
                pos_samples_list.append(neigh_gx)
        pos_samples[merge_m] = pos_samples_list

        # Generate negative samples using the existing neighbors from different sources
        neg_samples[merge_m] = gen_neg_samples_faster(neigh_g0, neigh_g1, neigh_g2)

    graph_dict["pos_samples"] = pos_samples
    graph_dict["neg_samples"] = neg_samples

    # Save pickled graph
    dill.dump(graph_dict, open(os.path.join(output_dir, salad_fname), "wb"))


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
