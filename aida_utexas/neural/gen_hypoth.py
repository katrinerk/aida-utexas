import json

import argparse

from aida_utexas.neural.modules import *
from aida_utexas.neural.utils import *
from aida_utexas import util

import torch.nn.functional as F
import sys
import os
import dill

def evaluate(seed_dir, indexed_data_dir, output_dir, model, device):
    for seed_path in sorted(seed_dir.glob('*_seeds.json')):
        print(f'Running model on cluster seeds from {seed_path}')

        with open(seed_path, 'r') as fin:
            seed_json = json.load(fin)

        sin_name = seed_path.stem.split('_')[0]
        indexed_data_dir_for_sin = indexed_data_dir / sin_name

        for idx, indexed_data_path in enumerate(sorted(indexed_data_dir_for_sin.glob('*.p'), key=lambda p: int(p.stem.split('_')[2]))):
            with open(indexed_data_path, 'rb') as fin:
                graph_dict = dill.load(fin)

            graph_dict['candidates'] = get_candidates(graph_dict)[0]

            curr_weight = -1

            i = 0

            add_stmts = []
            add_weights = []

            gcn_embeds = None

            while len([ere for ere in graph_dict['query_eres'] if graph_dict['graph_mix'].eres[graph_dict['ere_mat_ind'].get_word(ere)].category in ['Event', 'Relation']]) < 25:
                if len(graph_dict['candidates']) == 0:
                    break

                if i == 0:
                    _, coherence_out, gcn_embeds = model(graph_dict, None, device)
                else:
                    _, coherence_out, gcn_embeds = model(graph_dict, gcn_embeds, device)

                prediction = F.log_softmax(coherence_out, dim=0)
                predicted_index = select_valid_hypothesis(graph_dict, prediction)
                predicted_stmt_id = graph_dict['candidates'][predicted_index]
                add_stmts.append((graph_dict['stmt_mat_ind'].get_word(predicted_stmt_id)).split(graph_dict['graph_mix'].graph_id + '_')[-1])

                add_weights.append(curr_weight)
                curr_weight -= 1
                i += 1

                next_state(graph_dict, predicted_index)

            remove_duplicate_events(graph_dict)

            seed_json['support'][idx]['statements'] += add_stmts
            seed_json['support'][idx]['statementWeights'] += add_weights

        output_path = output_dir / (sin_name + '.json')
        with open(str(output_path), 'w') as fout:
            json.dump(seed_json, fout, indent=1)


def main():
    parser = argparse.ArgumentParser()

    # For qualitative evaluation only
    parser.add_argument('working_dir', help='path to the working directory')
    parser.add_argument('--seed_dir', default='cluster_seeds',
                        help='name of the subdirectory in working_dir containing cluster seeds')
    parser.add_argument('--indexed_data_dir', default='data_indexed',
                        help='name of the subdirectory in working_dir containing indexed data')
    parser.add_argument('--output_dir', default='result_jsons',
                        help='name of the subdirectory in working_dir to write output hypotheses')

    parser.add_argument("--indexer_path", default="resources/indexers.p",
                        help='path to the indexer file')
    parser.add_argument("--model_path", default="resources/gcn2-cuda_best_15000_0.ckpt",
                        help='path to the pre-trained model checkpoint')

    parser.add_argument("--device", type=int, default=-1)

    parser.add_argument("--attention_type", type=str, default='concat')
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--attention_size", type=int, default=300)
    parser.add_argument("--conv_dropout", type=float, default=.5)
    parser.add_argument("--attention_dropout", type=float, default=.3)

    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    # parser.add_argument("--use_highest_ranked_gold", action='store_true')
    # parser.add_argument("--force", action='store_true')
    # parser.add_argument("--init_prob_force", type=float, default=.95)
    # parser.add_argument("--force_decay", type=float, default=.9684)
    # parser.add_argument("--force_every", type=int, default=1000)
    # parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--self_attention_type", type=str, default='concat')
    # parser.add_argument("--num_epochs", type=int, default=2)
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    # parser.add_argument("--save_path", type=str, default="Saved_Models")
    # parser.add_argument("--save_tag", type=str, default="")
    # parser.add_argument("--eval_tag", type=str, default="")
    # parser.add_argument("--valid_every", type=int, default=5000)
    # parser.add_argument("--print_every", type=int, default=100)

    args = parser.parse_args()

    if args.device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    print(f'\nConfig:{args}')

    torch.manual_seed(0)
    np.random.seed(0)

    indexer_path = str(util.get_input_path(args.indexer_path))
    with open(indexer_path, 'rb') as fin:
        ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt = dill.load(fin)
    indexer_info_dict = dict()
    indexer_info_dict['ere_indexer'] = ere_indexer
    indexer_info_dict['stmt_indexer'] = stmt_indexer
    indexer_info_dict['ere_emb_mat'] = ere_emb_mat
    indexer_info_dict['stmt_emb_mat'] = stmt_emb_mat
    indexer_info_dict['num_word2vec_ere'] = num_word2vec_ere
    indexer_info_dict['num_word2vec_stmt'] = num_word2vec_stmt

    model = CoherenceNetWithGCN(
        False,
        indexer_info_dict,
        args.attention_type,
        None,
        args.num_layers,
        args.hidden_size,
        args.attention_size,
        args.conv_dropout,
        args.attention_dropout)

    model_path = str(util.get_input_path(args.model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])

    model.to(device)
    model.eval()

    working_dir = util.get_input_path(args.working_dir)
    seed_dir = util.get_input_path(working_dir / args.seed_dir)
    indexed_data_dir = util.get_input_path(working_dir / args.indexed_data_dir)
    output_dir = util.get_output_dir(working_dir / args.output_dir,
                                     overwrite_warning=not args.force)

    print('\nExpanding cluster seeds ...')

    evaluate(seed_dir, indexed_data_dir, output_dir, model, device)

    print(f'\nExpanding finished: raw hypotheses in directory {output_dir}')


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    main()
