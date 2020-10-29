# Original author: Su Wang, 2019
# Modified by Alex Tomkovich in 2019/2020

######
# This file contains the training pipeline for models trained on graph salads.
######

import dill
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
from copy import deepcopy
from modules import *

# Make a dir (if it doesn't already exist)
def verify_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# Run the model on a graph salad
def admit_seq(force, prob_force, model, optimizer, graph_dict, data_group, use_highest_ranked_gold, extraction_size, back_prop, device):
    graph_mix = graph_dict['graph_mix']
    target_graph_id = graph_dict['target_graph_id']
    # ere_mat_ind = graph_dict['ere_mat_ind']
    stmt_mat_ind = graph_dict['stmt_mat_ind']
    # query_eres = {ere_mat_ind.get_word(item) for item in graph_dict['query_eres']}
    query_stmts = {stmt_mat_ind.get_word(item) for item in graph_dict['query_stmts']}

    #
    # Find all mix points (points adjacent to statements from all three component graphs)
    # mix_points = {ere_id for ere_id in query_eres if len({graph_mix.stmts[stmt_id].graph_id for stmt_id in graph_mix.eres[ere_id].stmt_ids}) > 1}
    # seen_eres = set()
    # curr_eres = deepcopy(mix_points)
    #
    # # Traverse *via only target-graph stmts* the graph salad, starting at the mix points
    # while len(curr_eres) > 0:
    #     seen_eres.update(curr_eres)
    #
    #     for ere_id in deepcopy(curr_eres):
    #         target_neigh_eres = set.union(*[{graph_mix.stmts[stmt_id].head_id, graph_mix.stmts[stmt_id].tail_id} - {ere_id} for
    #                                         stmt_id in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[stmt_id].tail_id and graph_mix.stmts[stmt_id].graph_id == target_graph_id])
    #         curr_eres = set.union(curr_eres, target_neigh_eres) - seen_eres

    # Find all statements which are attached to any ERE reached via the above traversal (less the initial query stmts)
    #reachable_stmts = set.union(*[{item for item in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[item].graph_id == target_graph_id} for ere_id in seen_eres]) - query_stmts
    #assert reachable_stmts == {stmt_id for stmt_id in graph_mix.stmts.keys() if graph_mix.stmts[stmt_id].graph_id == target_graph_id} - query_stmts

    # Our extraction size is the minimum of
    # (i)  the number of target-graph stmts reachable from the mix points via only other target-graph stmts
    # (ii) the max number of extractions
    # (We don't want to ask our model to do 25 extractions if there are only 14 target-graph statements not already in the query set)
    num_target_graph_stmt_ids = len({stmt_id for stmt_id in graph_mix.stmts.keys() if graph_mix.stmts[stmt_id].graph_id == target_graph_id} - query_stmts)
    extraction_size = min(num_target_graph_stmt_ids, extraction_size)

    num_correct = 0
    predictions, trues = [], []

    # This should not be an issue anymore
    if extraction_size == 0:
        return False

    # Perform a sequence of candidate admissions
    for i in range(extraction_size):
        # For the first extraction, we need to obtain our GCN embeds; after that, we reuse this same set of embeds
        if i == 0:
            _, coherence_out, gcn_embeds = model(graph_dict, None, device)
        else:
            _, coherence_out, gcn_embeds = model(graph_dict, gcn_embeds, device)

        # Take a log softmax and identify the index of the highest-scoring valid candidate stmt
        prediction = F.log_softmax(coherence_out, dim=0)
        predicted_index = select_valid_hypothesis(graph_dict, prediction)#prediction.argmax().item()
        num_correct += 1 if graph_dict['stmt_class_labels'][predicted_index] == 1 else 0

        predictions.append(prediction.reshape(-1))
        trues.append(get_tensor_labels(graph_dict, device).reshape(-1))

        # If teacher forcing is on, draw a random number between (0, 1) and test against the current prob of teacher forcing
        # If the test succeeds, admit:
        # (i) a random target-graph candidate statement (if teacher forcing succeeds and a use_highest_ranked_gold is false)
        # (ii) the model's highest-ranked target-graph candidate statement (if teacher forcing succeeds and use_highest_ranked_gold is true)
        # (iii) the model's prediction, if teacher forcing is off/does not succeed
        if force and data_group == "Train":
            if np.random.uniform(0, 1) <= prob_force:
                random_true_index = get_random_gold_label(graph_dict, prediction, use_highest_ranked_gold)
                next_state(graph_dict, random_true_index)
            else:
                next_state(graph_dict, predicted_index)
        else:
            next_state(graph_dict, predicted_index)

    # Finish the full inference/extraction sequence, and then remove events that agree in event type and in the IDs of all arguments
    remove_duplicate_events(graph_dict)

    loss = multi_correct_nll_loss(predictions, trues, device)

    if data_group == "Train":
        loss.backward()

    # Only backprop when told to (as per the batch_size in the train() function)
    if data_group == "Train" and back_prop:
        optimizer.step()
        optimizer.zero_grad()

    # Accuracy (a misnomer for precision) is precision over extracted stmts
    # (i.e., of the <extraction_size> stmts admitted by the model, how many
    #  were actually from the target graph?)
    accuracy = float(num_correct) / extraction_size

    return loss.item(), accuracy

# Train the model on a set of graph salads
def train(batch_size, extraction_size, weight_decay, force, init_prob_force, force_decay, force_every, train_path, valid_path, test_path, indexer_info_dict, self_attend,
          attention_type, attn_head_stmt_tail, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout, num_epochs, learning_rate, save_path, load_path, load_optim,
          use_highest_ranked_gold, valid_every, print_every, device):
    model = CoherenceNetWithGCN(False, indexer_info_dict, attention_type, None, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout).to(device)

    # If a pretrained model should be used, load its parameters in
    if load_path is not None:
        model.load_state_dict(torch.load(load_path)['model'])

    # Instantiate Adam optimizer with specified weight decay (if any)
    if weight_decay:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load in an optimizer from a previous training run, if appropriate
    if load_path and load_optim:
        optimizer.load_state_dict(torch.load(load_path)['optimizer'])

    optimizer.zero_grad()

    start = time.time()

    current_epoch = 0

    best_loss = np.inf

    # Create DataIterator object to manage data
    train_iter = DataIterator(train_path)

    prob_force = init_prob_force

    while train_iter.epoch < num_epochs:
        step = 0
        print("\nEpoch %d\n" % (current_epoch + 1))
        model.train()
        train_losses, train_accuracies, train_diffs = [], [], []

        while train_iter.epoch <= current_epoch:
            # Every <force_every> graph instances, decay the prob of teacher forcing by a factor of <force_decay>
            if (step != 0 and step % force_every == 0) or (step == 0 and current_epoch > 0):
               prob_force *= force_decay

            # "Artificial" batching; allow gradients to accumulate and only backprop after every <batch_size> graph salads
            if (step != 0 and step % batch_size == 0) or (step == 0 and current_epoch > 0) or (step == len(train_iter.file_paths) - 1 and current_epoch == num_epochs - 1):
                result = admit_seq(force, prob_force, model, optimizer, train_iter.next_batch(), "Train", use_highest_ranked_gold, extraction_size, True, device)
            else:
                result = admit_seq(force, prob_force, model, optimizer, train_iter.next_batch(), "Train", use_highest_ranked_gold, extraction_size, False, device)

            if result: # All graph salads should be valid--we should be able to remove this if condition
                step += 1
                train_loss, train_accuracy = result
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

                # Print output report after every <print_every> graph salads
                if (step % print_every == 0) or (step == len(train_iter.file_paths)):
                    print("At step %d: loss = %.6f | accuracy = %.4f (%.2fs)." % (step, np.mean(train_losses), np.mean(train_accuracies), time.time() - start))
                    start = time.time()

                # Validate the model on the validation set after every <valid_every> salads
                if (step % valid_every == 0) or (step == len(train_iter.file_paths)):
                    print("\nRunning valid ...\n")
                    model.eval()
                    average_valid_loss, average_valid_accuracy = run_no_backprop(valid_path, extraction_size, model)
                    print("Valid (avg): loss = %.6f | accuracy = %.4f" % (average_valid_loss, average_valid_accuracy))

                    # Save checkpoint if the reported loss is lower than all previous reported losses
                    if average_valid_loss < best_loss:
                        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(save_path, 'gcn2-cuda_best' + '_' + str(step) + '_' + str(current_epoch) + '.ckpt'))
                        best_loss = average_valid_loss
                        print('New best val checkpoint at step ' + str(step) + ' of epoch ' + str(current_epoch + 1))

                    model.train()

        current_epoch += 1

    # Run the model on the test set
    model.eval()
    average_test_loss, average_test_accuracy = run_no_backprop(test_path, extraction_size, model)
    print("Test (avg): loss = %.6f | accuracy = %.4f" % (average_test_loss, average_test_accuracy))
    model.train()

# Run the model on a set of data for evaluative purposes (called when validating and testing only)
def run_no_backprop(data_path, extraction_size, model):
    valid_iter = DataIterator(data_path)

    valid_losses, valid_accuracies = [], []

    while valid_iter.epoch == 0:
        result = admit_seq(False, None, model, None, valid_iter.next_batch(), "Val", False, extraction_size, False, device)
        if result:
            valid_loss, valid_accuracy = result
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

    average_valid_loss = np.mean(valid_losses)
    average_valid_accuracy = np.mean(valid_accuracies)

    return (average_valid_loss, average_valid_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        help="Mode to run script in (can be either train or validate)")
    parser.add_argument("--data_dir", type=str, default="/home/atomko/M36_50k_Indexed",
                        help="Data directory")
    parser.add_argument("--train_dir", type=str, default="Train",
                        help="Name of subdir under data directory containing training mixtures")
    parser.add_argument("--valid_dir", type=str, default="Val_1k",
                        help="Name of subdir under data directory containing validation mixtures")
    parser.add_argument("--test_dir", type=str, default="Test",
                        help="Name of subdir under data directory containing test mixtures")
    parser.add_argument("--indexer_info_file", type=str, default="/home/atomko/M36_50k_Indexed/indexers.p",
                        help="Indexer info file (contains word-to-index maps for ERE and statement names, contains top k most frequent Word2Vec embeds in training set)")
    parser.add_argument("--use_highest_ranked_gold", action='store_true',
                        help="During teacher forcing, always provide the model with the target-narrative candidate which was ranked most highly by the model")
    parser.add_argument("--force", action='store_true',
                        help="Enable teacher forcing")
    parser.add_argument("--init_prob_force", type=float, default=.95,
                        help="Initial probability of teacher forcing (i.e., admitting a target-narrative candidate when the model's highest-ranked candidate is incorrect)")
    parser.add_argument("--force_decay", type=float, default=.9607,
                        help="Factor by which the current probability of teacher forcing is multiplied every <force_every> training instances")
    parser.add_argument("--force_every", type=int, default=2500,
                        help="The current prob of teacher forcing is multiplied by the factor <force_decay> every <force_every> training instances")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Number of graph salads per batch")
    parser.add_argument("--extraction_size", type=int, default=25,
                        help="Maximum number of statements to admit per graph salad")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers in the GCN")
    parser.add_argument("--attention_type", type=str, default='concat',
                        help="Attention score type for inference (can be concat or bilinear)")
    parser.add_argument("--self_attend", action='store_true',
                        help="Enable a self-attention mechanism during convolution")
    parser.add_argument("--self_attention_type", type=str, default='concat',
                        help="Attention score type for self-attention (can be concat or bilinear)")
    parser.add_argument("--hidden_size", type=int, default=300,
                        help="Size of hidden representations in model")
    parser.add_argument("--attention_size", type=int, default=300,
                        help="Size of attention vectors in model")
    parser.add_argument("--conv_dropout", type=float, default=.5,
                        help="Dropout rate for convolution layers")
    parser.add_argument("--attention_dropout", type=float, default=.3,
                        help="Dropout rate for attention layers")
    parser.add_argument("--attn_head_stmt_tail", action='store_true',
                        help="Enable <head rep><statement rep><tail rep> concatenated representations")
    parser.add_argument("--num_epochs", type=int, default=4,
                        help="Number of epochs for training")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay for Adam optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--save_path", type=str, default="/home/atomko/M36_Saved_Models",
                        help="Directory to which model/optimizer checkpoints will be saved")
    parser.add_argument("--load_path", type=str, default=None,
                        help="File path for pretrained model/optimizer")
    parser.add_argument("--load_optim", action='store_true',
                        help="Load optimizer state (if <load_path> is also supplied)")
    parser.add_argument("--valid_every", type=int, default=5000,
                        help="Model runs on the validation set every <valid_every> training instances")
    parser.add_argument("--print_every", type=int, default=50,
                        help="Precision and loss reports are generated every <print_every> training instances")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device number (or 'cpu' for cpu)")

    args = parser.parse_args()
    locals().update(vars(args))
    load_path = None if load_path == "" else load_path
    device = torch.device(("cuda:" + device) if torch.cuda.is_available() and device != "cpu" else "cpu")

    print("Config:\n", args, "\nDevice:\n", device, "\n")
    print("START TRAINING ...\n\n")

    # Check that the directory for saving model checkpoints exists
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

    # Train a fresh model
    if mode == 'train':
        train(batch_size, extraction_size, weight_decay, force, init_prob_force, force_decay, force_every, train_path, valid_path, test_path, indexer_info_dict, self_attend, attention_type, attn_head_stmt_tail, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout, num_epochs,
                      learning_rate, save_path, load_path, load_optim, use_highest_ranked_gold, valid_every, print_every, device)
    # Evaluate an existing model on test data
    elif mode == 'validate':
        model = CoherenceNetWithGCN(indexer_info_dict, self_attend, attention_type, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout).to(device)
        model.load_state_dict(torch.load(load_path, map_location=device)['model'])

        print("\nRunning on test set ...\n")
        model.eval()
        average_valid_loss, average_valid_accuracy = run_no_backprop(test_path, extraction_size, model)
        print("Test (avg): loss = %.6f | accuracy = %.4f" % (average_valid_loss, average_valid_accuracy))
        model.train()