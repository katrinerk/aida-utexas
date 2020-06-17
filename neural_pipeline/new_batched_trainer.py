import dill
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import re

from new_batched_modules import *


def run_batch(force, prob_force, model, optimizer, batch, data_group, use_highest_ranked_gold, extraction_size, device):
    total_golds_list = [len(set([batch[iter]['stmt_mat_ind'].get_index(stmt_id, add=False) for stmt_id in batch[iter]['graph_mix'].stmts.keys() if batch[iter]['graph_mix'].stmts[stmt_id].graph_id == batch[iter]['target_graph_id']]) - set(batch[iter]['query_stmts'])) for iter in range(len(batch))]
    extraction_size_list = [min(total_golds_list[iter], extraction_size) for iter in range(len(batch))]
    orig_extraction_size_sum = sum(extraction_size_list)

    num_correct = 0
    predictions, trues = [], []

    if 0 in extraction_size_list:##TO-DO
        return False##TO-DO

    for i in range(max(extraction_size_list)):
        keep_list = [iter for iter in range(len(batch)) if extraction_size_list[iter] > i]

        batch = [item for (iter, item) in enumerate(batch) if iter in keep_list]
        extraction_size_list = [item for (iter, item) in enumerate(extraction_size_list) if iter in keep_list]

        padded_batch_mats = reset_batch_mats(batch, device)

        if i == 0:
            _, _, prediction_list, gcn_embeds = model(batch, padded_batch_mats, None, device)
        else:
            gcn_embeds['eres'] = gcn_embeds['eres'][keep_list, :padded_batch_mats['adj_head'].shape[1], :]
            gcn_embeds['stmts'] = gcn_embeds['stmts'][keep_list, :padded_batch_mats['adj_head'].shape[2], :]

            _, _, prediction_list, gcn_embeds = model(batch, padded_batch_mats, gcn_embeds, device)

        predicted_indices = [prediction_list[iter].argmax().item() for iter in range(len(batch))]

        for iter, predicted_index in enumerate(predicted_indices):
            num_correct += 1 if batch[iter]['label_inputs'][predicted_index] == 1 else 0

        [predictions.append(prediction_list[iter].reshape(-1)[:len(batch[iter]['candidates'])]) for iter in range(len(batch))]
        [trues.append(item.reshape(-1)) for item in get_tensor_labels(batch, device)]

        if force and data_group == "Train":
            if np.random.uniform(0, 1) <= prob_force:
                random_true_indices = get_random_gold_label(batch, prediction_list, use_highest_ranked_gold)
                next_state(batch, random_true_indices, False)
            else:
                next_state(batch, predicted_indices, False)
        else:
            next_state(batch, predicted_indices, False)

    loss = multi_correct_nll_loss(predictions, trues, device)

    if data_group == "Train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = float(num_correct) / orig_extraction_size_sum

    return loss.item(), accuracy

def reset_batch_mats(batch, device):
    max_num_eres = max([len(batch[iter]['ere_labels']) for iter in range(len(batch))])
    max_num_stmts = max([len(batch[iter]['stmt_labels']) for iter in range(len(batch))])

    padded_batch_mats = {}
    padded_batch_mats['adj_head'] = torch.zeros((len(batch), max_num_eres, max_num_stmts)).to(device=device)
    padded_batch_mats['adj_tail'] = torch.zeros((len(batch), max_num_eres, max_num_stmts)).to(device=device)
    padded_batch_mats['adj_type'] = torch.zeros((len(batch), max_num_eres, max_num_stmts)).to(device=device)
    #padded_batch_mats['inv_sqrt_degree_eres'] = torch.zeros((len(batch), max_num_eres, max_num_eres)).to(device=device)
    #padded_batch_mats['inv_sqrt_degree_stmts'] = torch.zeros((len(batch), max_num_stmts, max_num_stmts)).to(device=device)

    for iter in range(len(batch)):
        #batch[iter]['inv_sqrt_degree_eres'] = torch.sqrt((1 / (1 + torch.sum(batch[iter]['adj_head'], dim=1) + torch.sum(batch[iter]['adj_tail'], dim=1) + torch.sum(batch[iter]['adj_type'], dim=1))) * torch.eye(len(batch[iter]['ere_labels'])))
        #batch[iter]['inv_sqrt_degree_stmts'] = torch.sqrt((1 / (1 + torch.sum(torch.transpose(batch[iter]['adj_head'], 0, 1), dim=1) + torch.sum(torch.transpose(batch[iter]['adj_tail'], 0, 1), dim=1) + torch.sum(torch.transpose(batch[iter]['adj_type'], 0, 1), dim=1))) * torch.eye(len(batch[iter]['stmt_labels'])))
        padded_batch_mats['adj_head'][iter][:batch[iter]['adj_head'].shape[0], :batch[iter]['adj_head'].shape[1]] = batch[iter]['adj_head']
        padded_batch_mats['adj_tail'][iter][:batch[iter]['adj_tail'].shape[0], :batch[iter]['adj_tail'].shape[1]] = batch[iter]['adj_tail']
        padded_batch_mats['adj_type'][iter][:batch[iter]['adj_type'].shape[0], :batch[iter]['adj_type'].shape[1]] = batch[iter]['adj_type']
        #padded_batch_mats['inv_sqrt_degree_eres'][iter][:batch[iter]['inv_sqrt_degree_eres'].shape[0], :batch[iter]['inv_sqrt_degree_eres'].shape[1]] = batch[iter]['inv_sqrt_degree_eres']
        #padded_batch_mats['inv_sqrt_degree_stmts'][iter][:batch[iter]['inv_sqrt_degree_stmts'].shape[0], :batch[iter]['inv_sqrt_degree_stmts'].shape[1]] = batch[iter]['inv_sqrt_degree_stmts']

    return padded_batch_mats


def train(batch_size, extraction_size, force, init_prob_force, force_decay, force_every, train_path, valid_path, test_path, indexer_info_dict, self_attend,
          attention_type, attn_head_stmt_tail, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout, num_epochs, learning_rate, save_path, load_path, load_optim,
          use_highest_ranked_gold, valid_every, print_every, device):
    model = CoherenceNetWithGCN(indexer_info_dict, self_attend, attention_type, attn_head_stmt_tail, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout).to(device)

    if load_path is not None:
        model.load_state_dict(torch.load(load_path)['model'])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.5)

    if load_path and load_optim:
        optimizer.load_state_dict(torch.load(load_path)['optimizer'])

    start = time.time()

    current_epoch = 0

    train_iter = DataIterator(train_path, batch_size)

    prob_force = init_prob_force##TO-DO

    while train_iter.epoch < num_epochs:
        step = 0
        print("\nEpoch %d\n" % (current_epoch + 1))
        model.train()
        train_losses, train_accuracies, train_diffs = [], [], []
        while train_iter.epoch <= current_epoch:
            if step != 0 and step % force_every == 0:##TO-DO
               prob_force *= force_decay##TO-DO

            result = run_batch(force, prob_force, model, optimizer, train_iter.next_batch(), "Train", use_highest_ranked_gold, extraction_size, device)

            if result:#TO-DO
                step += batch_size
                train_loss, train_accuracy = result
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

                if step % print_every == 0:
                    print("At step %d: loss = %.6f | accuracy = %.4f (%.2fs)." % (step, np.mean(train_losses), np.mean(train_accuracies), time.time() - start))
                    start = time.time()

                if step % valid_every == 0:
                    print("\nRunning valid ...\n")
                    print("Train (avg) loss = %.6f | accuracy = %.4f" % (np.mean(train_losses), np.mean(train_accuracies)))

                    model.eval()
                    average_valid_loss, average_valid_accuracy = run_no_backprop(valid_path, batch_size, extraction_size, model)
                    print("Valid (avg): loss = %.6f | accuracy = %.4f" % (average_valid_loss, average_valid_accuracy))
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(save_path, 'gcn2-cuda_best' + '_' + str(step) + '_' + str(current_epoch) + '.ckpt'))
                    model.train()

        model.eval()
        average_test_loss, average_test_accuracy = run_no_backprop(test_path, batch_size, extraction_size, model)
        print("Test (avg): loss = %.6f | accuracy = %.4f" % (average_test_loss, average_test_accuracy))
        model.train()

        current_epoch += 1


def run_no_backprop(data_path, batch_size, extraction_size, model):
    valid_iter = DataIterator(data_path, batch_size)

    valid_losses, valid_accuracies = [], []
    while valid_iter.epoch == 0:
        result = run_batch(False, None, model, None, valid_iter.next_batch(), "Val", False, extraction_size, device)
        if result:#TO-DO
            valid_loss, valid_accuracy = result
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

    average_valid_loss = np.mean(valid_losses)
    average_valid_accuracy = np.mean(valid_accuracies)

    return (average_valid_loss, average_valid_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Mode to run script in (can be either train or eval)")
    parser.add_argument("--data_dir", type=str, default="/home/atomko/backup_drive/AIDA_Data_Gen_10_6/Mixtures_Indexed_Pretrain", help="Data directory")
    parser.add_argument("--train_dir", type=str, default="Train", help="Name of subdir under data directory containing training mixtures")
    parser.add_argument("--valid_dir", type=str, default="Val", help="Name of subdir under data directory containing validation mixtures")
    parser.add_argument("--test_dir", type=str, default="Test", help="Name of subdir under data directory containing test mixtures")
    parser.add_argument("--indexer_info_file", type=str, default="indexers.p", help="Indexer info file (contains word-to-index maps for ERE and statement names, contains top k most frequent Word2Vec embeds in training set)")
    parser.add_argument("--use_highest_ranked_gold", action='store_true', help="During teacher forcing, always provide the model with the target-narrative candidate which was ranked most highly by the model")
    parser.add_argument("--force", action='store_true', help="Enable teacher forcing")
    parser.add_argument("--init_prob_force", type=float, default=.95, help="Initial probability of admitting a target-narrative candidate when the model's highest-ranked candidate is incorrect")
    parser.add_argument("--force_decay", type=float, default=.9684, help="Factor by which the current probability of teacher forcing is multiplied every <force_every> training instances")
    parser.add_argument("--force_every", type=int, default=1000, help="The current prob of teacher forcing is multiplied by the factor <force_decay> every <force_every> training instances")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of graph salads the model processes at a time")
    parser.add_argument("--extraction_size", type=int, default=25, help="Maximum number of statements to admit per graph instance")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the GCN")
    parser.add_argument("--attention_type", type=str, default='concat', help="Attention score type for inference (can be concat or bilinear)")
    parser.add_argument("--self_attend", action='store_true', help="Enable a self-attention mechanism during convolution")
    parser.add_argument("--self_attention_type", type=str, default='concat', help="Attention score type for convolution (can be concat or bilinear)")
    parser.add_argument("--hidden_size", type=int, default=300, help="Size of hidden representations in model")
    parser.add_argument("--attention_size", type=int, default=300, help="Size of attention vectors in model")
    parser.add_argument("--conv_dropout", type=float, default=.5, help="Dropout rate for convolution layers")
    parser.add_argument("--attention_dropout", type=float, default=.2, help="Dropout rate for attention layers")
    parser.add_argument("--attn_head_stmt_tail", action='store_true', help="Enable <head rep><statement rep><tail rep> concatenated representations")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="SavedModels", help="Directory to which model/optimizer checkpoints will be saved")
    parser.add_argument("--load_path", type=str, default=None, help="File path for pretrained model/optimizer")
    parser.add_argument("--load_optim", action='store_true', help="Load optimizer state (if <load_path> is also supplied)")
    parser.add_argument("--valid_every", type=int, default=20000, help="Model runs on the validation set every <valid_every> training instances")
    parser.add_argument("--print_every", type=int, default=100, help="Precision and loss reports are generated every <print_every> training instances")
    parser.add_argument("--device", type=str, default="0", help="CUDA device number (or 'cpu' for cpu)")

    args = parser.parse_args()
    locals().update(vars(args))
    load_path = None if load_path == "" else load_path
    device = torch.device(("cuda:" + device) if torch.cuda.is_available() and device != "cpu" else "cpu")

    print("Config:\n", args, "\nDevice:\n", device, "\n")
    print("START TRAINING ...\n\n")

    verify_dir(save_path)

    # Set seeds for debugging/hyperparam comparison
    torch.manual_seed(0)
    np.random.seed(0)

    train_path = os.path.join(data_dir, train_dir)
    valid_path = os.path.join(data_dir, valid_dir)
    test_path = os.path.join(data_dir, test_dir)

    # Load indexer info
    indexer_info_dict = dict()
    ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt = dill.load(open(indexer_info_file, 'rb'))
    indexer_info_dict['ere_indexer'] = ere_indexer
    indexer_info_dict['stmt_indexer'] = stmt_indexer
    indexer_info_dict['ere_emb_mat'] = ere_emb_mat
    indexer_info_dict['stmt_emb_mat'] = stmt_emb_mat
    indexer_info_dict['num_word2vec_ere'] = num_word2vec_ere
    indexer_info_dict['num_word2vec_stmt'] = num_word2vec_stmt

    if mode == 'train':
        train(batch_size, extraction_size, force, init_prob_force, force_decay, force_every, train_path, valid_path, test_path, indexer_info_dict, self_attend, attention_type, attn_head_stmt_tail, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout, num_epochs,
                      learning_rate, save_path, load_path, load_optim, use_highest_ranked_gold, valid_every, print_every, device)
    else:
        model = CoherenceNetWithGCN(ere_embed_size, ere_indexer.size, stmt_embed_size, stmt_indexer.size, attention_type, attn_head_stmt_tail, num_layers, hidden_size, attention_size, .1).to(device)
        model.load_state_dict(torch.load(load_path, map_location={'cuda:1': 'cpu', 'cuda:0': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu'})['model'])

        if mode == 'validate':
            print("\nRunning valid ...\n")
            model.eval()
            [validate(batch_size, item, ere_indexer, stmt_indexer, len(os.listdir(item)), model) for item in val_sets]
            [validate(batch_size, item, ere_indexer, stmt_indexer, len(os.listdir(item)), model) for item in test_sets]
            model.train()
        else:
            model.eval()
            evaluate(eval_tag, ldc_gold_dir, output_dir, model)
            model.train()