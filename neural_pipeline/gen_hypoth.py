import dill
import math
import numpy as np
import time
import torch
from torch.autograd import Variable as Var
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import re

from modules import *
from utils import *

def reset_batch_mats(batch, device):
    max_num_eres = max([len(batch[iter]['ere_labels']) for iter in range(len(batch))])
    max_num_stmts = max([len(batch[iter]['stmt_labels']) for iter in range(len(batch))])

    padded_batch_mats = {}
    padded_batch_mats['adj_head'] = torch.zeros((len(batch), max_num_eres, max_num_stmts)).to(device=device)
    padded_batch_mats['adj_tail'] = torch.zeros((len(batch), max_num_eres, max_num_stmts)).to(device=device)
    padded_batch_mats['adj_type'] = torch.zeros((len(batch), max_num_eres, max_num_stmts)).to(device=device)

    for iter in range(len(batch)):
        padded_batch_mats['adj_head'][iter][:batch[iter]['adj_head'].shape[0], :batch[iter]['adj_head'].shape[1]] = batch[iter]['adj_head']
        padded_batch_mats['adj_tail'][iter][:batch[iter]['adj_tail'].shape[0], :batch[iter]['adj_tail'].shape[1]] = batch[iter]['adj_tail']
        padded_batch_mats['adj_type'][iter][:batch[iter]['adj_type'].shape[0], :batch[iter]['adj_type'].shape[1]] = batch[iter]['adj_type']

    return padded_batch_mats

def evaluate(seed_dir, hypothesis_dir_name, eval_data_dir, output_dir, model):
    for group_name in os.listdir(eval_data_dir):
        if not os.path.exists(os.path.join(output_dir, group_name)):
            os.makedirs(os.path.join(output_dir, group_name))

            if not os.path.exists(os.path.join(output_dir, group_name, 'result_jsons')):
                os.makedirs(os.path.join(output_dir, group_name, 'result_jsons'))

        for seed_num in os.listdir(os.path.join(eval_data_dir, group_name)):
            print('Running model on cluster ' + seed_num + ' for group ' + group_name + '------')
            seed_path = os.path.join(seed_dir, group_name, hypothesis_dir_name, seed_num + '_seeds.json')

            seed_json = json.load(open(seed_path, 'r')).copy()

            for subgraph_iter, subgraph in enumerate(sorted(os.listdir(os.path.join(eval_data_dir, group_name, seed_num)), key=lambda x : int(x.split('_')[-1].split('.p')[0]))):
                batch = [dill.load(open(os.path.join(eval_data_dir, group_name, seed_num, subgraph), 'rb'))]

                batch[0]['candidates'] = get_candidates(batch)[0]

                curr_weight = -1

                i = 0

                add_stmts = []
                add_weights = []

                while len([ere for ere in batch[0]['query_eres'] if batch[0]['graph_mix'].eres[batch[0]['ere_mat_ind'].get_word(ere)].category in ['Event', 'Relation']]) < 25:
                    padded_batch_mats = reset_batch_mats(batch, device)
                    if i == 0:
                        prediction, gcn_embeds = model(batch, padded_batch_mats, None, device)
                    else:
                        prediction, gcn_embeds = model(batch, padded_batch_mats, gcn_embeds, device)

                    predicted_index = prediction.argmax().item()
                    predicted_stmt_id = batch[0]['candidates'][predicted_index]
                    add_stmts.append((batch[0]['stmt_mat_ind'].get_word(predicted_stmt_id)).split(batch[0]['graph_mix'].graph_id + '_')[-1])

                    add_weights.append(curr_weight)
                    curr_weight -= 1
                    i += 1

                    next_state(batch, [predicted_index], True)

                seed_json['support'][subgraph_iter]['statements'] += add_stmts
                seed_json['support'][subgraph_iter]['statementWeights'] += add_weights


            json.dump(seed_json, open(os.path.join(output_dir, group_name, 'result_jsons', seed_num + '_seeds.json'), 'w'), indent=1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # For qualitative evaluation only
    parser.add_argument("--seed_dir", type=str, default="/aida-utexas/data")
    parser.add_argument("--hypothesis_dir_name", type=str, default="cluster_seeds")
    parser.add_argument("--eval_data_dir", type=str, default="/aida-utexas/data_indexed")
    parser.add_argument("--output_dir", type=str, default="/aida-utexas/hypoth_out")

    parser.add_argument("--indexer_info_file", type=str, default="/aida-utexas/neural_pipeline/indexers.p")

    parser.add_argument("--use_highest_ranked_gold", action='store_true')
    parser.add_argument("--force", action='store_true')
    parser.add_argument("--init_prob_force", type=float, default=.95)
    parser.add_argument("--force_decay", type=float, default=.9684)
    parser.add_argument("--force_every", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--attention_type", type=str, default='concat')
    parser.add_argument("--self_attend", action='store_true')
    parser.add_argument("--self_attention_type", type=str, default='concat')
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--attention_size", type=int, default=300)
    parser.add_argument("--conv_dropout", type=float, default=.5)
    parser.add_argument("--attention_dropout", type=float, default=.2)
    parser.add_argument("--attn_head_stmt_tail", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_path", type=str, default="Saved_Models")
    parser.add_argument("--save_tag", type=str, default="")
    parser.add_argument("--eval_tag", type=str, default="")
    parser.add_argument("--load_path", type=str, default="/aida-utexas/neural_pipeline/gcn2-cuda_best_5000_1.ckpt")
    parser.add_argument("--valid_every", type=int, default=5000)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()
    locals().update(vars(args))
    load_path = None if load_path == "" else load_path
    device = torch.device("cpu")

    print("Config:\n", args, "\nDevice:\n", device, "\n")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.manual_seed(0)
    np.random.seed(0)

    wiki_wiki_train_path = None
    wiki_aida_train_path = None
    aida_aida_train_path = None

    wiki_wiki_valid_path = None
    wiki_aida_valid_path = None
    aida_aida_valid_path = None

    wiki_wiki_test_path = None
    wiki_aida_test_path = None
    aida_aida_test_path = None

    indexer_info_dict = dict()
    ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt = dill.load(open(indexer_info_file, 'rb'))
    indexer_info_dict['ere_indexer'] = ere_indexer
    indexer_info_dict['stmt_indexer'] = stmt_indexer
    indexer_info_dict['ere_emb_mat'] = ere_emb_mat
    indexer_info_dict['stmt_emb_mat'] = stmt_emb_mat
    indexer_info_dict['num_word2vec_ere'] = num_word2vec_ere
    indexer_info_dict['num_word2vec_stmt'] = num_word2vec_stmt

    model = CoherenceNetWithGCN(indexer_info_dict, self_attend, attention_type, attn_head_stmt_tail, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout).to(device)
    model.load_state_dict(torch.load(load_path, map_location={'cuda:1': 'cpu', 'cuda:0': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu'})['model'])

    model.eval()
    evaluate(seed_dir, hypothesis_dir_name, eval_data_dir, output_dir, model)