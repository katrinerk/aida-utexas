""" PyTorch modules.

Author: Su Wang; 2019.
"""

import math

import torch.nn as nn
import torch.nn.functional as F

from aida_utexas.neural.utils import *


class Embeddings(nn.Module):
    """Embedding lookup."""

    def __init__(self, embed_size, vocab_size):
        """
        Args:
            embed_size: embedding size.
            vocab_size: vocab_size size.
        """
        super(Embeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, inputs):
        """
        Args:
            inputs: <batch-size, seq-length>.
        """
        # Lookup embeddings: <batch-size, seq-length>
        #                 -> <batch-size, seq-length, embed-size>
        return self.embeddings(inputs) * math.sqrt(self.embed_size)


class Score(nn.Module):
    """Blinear layer."""

    def __init__(self, attention_type, hidden_size):
        """Initializer.

        Args:
            hidden_size: the size of bilinear (square) weights.
        """
        super(Score, self).__init__()

        self.attention_type = attention_type

        if attention_type == 'bilinear':
            self.bilinear = nn.Linear(hidden_size, hidden_size)
            torch.nn.init.xavier_uniform_(self.bilinear.weight)

        elif attention_type == 'concat':
            self.concat = nn.Linear(hidden_size * 2, hidden_size)
            torch.nn.init.xavier_uniform_(self.concat.weight)
            self.single = nn.Linear(hidden_size, 1)
            torch.nn.init.xavier_uniform_(self.single.weight)

    def forward(self, first_inputs, second_inputs):
        """Forwarding.

        Args:
            first_inputs: feature matrix of shape <batch-size, hidden-size>.
            second_inputs: another feature matrix with the same shape as `first_inputs`.
        Returns:
            Bilinear (binary) interaction matrix of shape <batch-size, batch-size>.
        """
        if self.attention_type == 'bilinear':
            return self.bilinear(first_inputs).bmm(second_inputs.transpose(1, 2))
        elif self.attention_type == 'concat':
            num_attendees = first_inputs.size(1)
            num_attenders = second_inputs.size(1)
            concat_inputs = torch.cat([first_inputs.unsqueeze(2).expand(-1, -1, num_attenders, -1), second_inputs.unsqueeze(1).expand(-1, num_attendees, -1, -1)], dim=-1)
            return self.single(torch.tanh(self.concat(concat_inputs))).squeeze(-1)

class Attention(nn.Module):
    """Luong attention.

    Source: https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
    """

    def __init__(self, attn_head_stmt_tail, attention_type, hidden_size, attention_size, attention_dropout):
        """Initializer.

        Args:
            hidden_size: size of bi-linear interaction weights.
            attention_size: size of attention vectors.
        """
        super(Attention, self).__init__()

        if attn_head_stmt_tail:
            self.concat_trans = nn.Linear(hidden_size * 3, hidden_size)
            self.score = Score(attention_type, hidden_size)
            self.linear = nn.Linear(hidden_size * 2, attention_size)
            torch.nn.init.xavier_uniform_(self.concat_trans.weight)
        else:
            self.linear = nn.Linear(hidden_size * 3, attention_size)
            self.score_stmt_to_stmt = Score(attention_type, hidden_size)
            self.score_ere_to_stmt = Score(attention_type, hidden_size)

        self.att_dropout = torch.nn.Dropout(p=attention_dropout, inplace=False)
        self.attn_head_stmt_tail = attn_head_stmt_tail

        torch.nn.init.xavier_uniform_(self.linear.weight)

    def get_attention_weights(self, attendee_stmts, attendee_eres, attender, mask_stmt_to_stmt, mask_ere_to_stmt):
        """Compute attention weights (Eq (1) in source)."""
        if self.attn_head_stmt_tail:
            unnorm_attention_weights = self.score(attendee_stmts, attender)

            unnorm_attention_weights = torch.Tensor.masked_fill(unnorm_attention_weights, mask_stmt_to_stmt, float('-inf'))

            attention_weights = F.softmax(unnorm_attention_weights, dim=1)

            attention_weights = self.attn_dropout(attention_weights)

            attention_weights = torch.Tensor.masked_fill(attention_weights, mask_stmt_to_stmt, 0)
        else:
            unnorm_attention_weights_stmt_to_stmt = self.score_stmt_to_stmt(attendee_stmts, attender)
            unnorm_attention_weights_ere_to_stmt = self.score_ere_to_stmt(attendee_eres, attender)

            unnorm_attention_weights_stmt_to_stmt = torch.Tensor.masked_fill(unnorm_attention_weights_stmt_to_stmt, mask_stmt_to_stmt, float('-inf'))
            unnorm_attention_weights_ere_to_stmt = torch.Tensor.masked_fill(unnorm_attention_weights_ere_to_stmt, mask_ere_to_stmt, float('-inf'))

            attention_weights_stmt_to_stmt = F.softmax(unnorm_attention_weights_stmt_to_stmt, dim=1)
            attention_weights_ere_to_stmt = F.softmax(unnorm_attention_weights_ere_to_stmt, dim=1)

            attention_weights_stmt_to_stmt = self.att_dropout(attention_weights_stmt_to_stmt)
            attention_weights_ere_to_stmt = self.att_dropout(attention_weights_ere_to_stmt)

            attention_weights_stmt_to_stmt = torch.Tensor.masked_fill(attention_weights_stmt_to_stmt, mask_stmt_to_stmt, 0)
            attention_weights_ere_to_stmt = torch.Tensor.masked_fill(attention_weights_ere_to_stmt, mask_ere_to_stmt, 0)

            attention_weights = (attention_weights_stmt_to_stmt, attention_weights_ere_to_stmt)
        return attention_weights

    def get_context_vectors(self, attendee, attention_weights):
        if self.attn_head_stmt_tail:
            return torch.transpose(attention_weights, 1, 2).bmm(attendee)
        else:
            context_vectors_stmt_to_stmt = torch.transpose(attention_weights[0], 1, 2).bmm(attendee[0])
            context_vectors_ere_to_stmt = torch.transpose(attention_weights[1], 1, 2).bmm(attendee[1])

            return (context_vectors_stmt_to_stmt, context_vectors_ere_to_stmt)

    def get_attention_vectors(self, batch, padded_batch_mats, gcn_embeds, attendee_stmts, attendee_eres, attender, mask_stmt_to_stmt, mask_ere_to_stmt, device):
        if self.attn_head_stmt_tail:
            adj_head = padded_batch_mats['adj_head']
            adj_tail = padded_batch_mats['adj_tail']

            concat_attendees = torch.zeros((len(batch), attendee_stmts.shape[1], attendee_eres.shape[2] + attendee_stmts.shape[2] + attendee_eres.shape[2])).to(device=device)
            concat_attenders = torch.zeros((len(batch), attender.shape[1], attendee_eres.shape[2] + attendee_stmts.shape[2] + attendee_eres.shape[2])).to(device=device)

            for iter in range(len(batch)):
                concat_attendees[iter][:len(batch[iter]['query_stmts']), :] = torch.cat((torch.transpose(adj_head[iter], 0, 1).to(device=device).mm(gcn_embeds['eres'][iter])[list(batch[iter]['query_stmts'])], attendee_stmts[iter][:len(batch[iter]['query_stmts'])], torch.transpose(adj_tail[iter], 0, 1).to(device=device).mm(gcn_embeds['eres'][iter])[list(batch[iter]['query_stmts'])]), dim=-1)
                concat_attenders[iter][:len(batch[iter]['candidates']), :] = torch.cat((torch.transpose(adj_head[iter], 0, 1).to(device=device).mm(gcn_embeds['eres'][iter])[list(batch[iter]['candidates'])], attender[iter][:len(batch[iter]['candidates'])], torch.transpose(adj_tail[iter], 0, 1).to(device=device).mm(gcn_embeds['eres'][iter])[list(batch[iter]['candidates'])]), dim=-1)

            trans_attendees = torch.tanh(self.concat_trans(concat_attendees))
            trans_attenders = torch.tanh(self.concat_trans(concat_attenders))

            attention_weights = self.get_attention_weights(trans_attendees, None, trans_attenders, mask_stmt_to_stmt, None)
            context_vectors = self.get_context_vectors(trans_attendees, attention_weights)
            attention_vectors = torch.tanh(self.linear(torch.cat([trans_attenders, context_vectors], dim=-1)))
        else:
            attention_weights = self.get_attention_weights(attendee_stmts, attendee_eres, attender, mask_stmt_to_stmt, mask_ere_to_stmt)
            context_vectors = self.get_context_vectors((attendee_stmts, attendee_eres), attention_weights)
            attention_vectors = torch.tanh(self.linear(torch.cat([attender, context_vectors[0], context_vectors[1]], dim=-1)))
        return attention_vectors

def pad(indices):
    """Pad a list of lists to the longest sublist. Return as a list of lists (easy manip. in run_batch)."""
    padded_indices = []
    max_len = max(len(sub_indices) for sub_indices in indices)
    padded_indices = [sub_indices[:max_len]
                      if len(sub_indices) >= max_len
                      else sub_indices + [0] * (max_len - len(sub_indices))
                      for sub_indices in indices]
    return padded_indices


def to_tensor(inputs, tensor_type=torch.LongTensor, device=torch.device("cpu")):
    return tensor_type(np.array(inputs)).to(device)


class CoherenceNetWithGCN(nn.Module):

    def __init__(self, indexer_info_dict, self_attend, attention_type, attn_head_stmt_tail, num_layers, hidden_size, attention_size, conv_dropout, attention_dropout):
        super(CoherenceNetWithGCN, self).__init__()
        ere_emb = indexer_info_dict['ere_emb_mat']
        stmt_emb = indexer_info_dict['stmt_emb_mat']
        ere_embed_size = ere_emb.shape[1]
        stmt_embed_size = stmt_emb.shape[1]
        self.ere_embedder = nn.Embedding.from_pretrained(torch.from_numpy(ere_emb), freeze=False)
        self.stmt_embedder = nn.Embedding.from_pretrained(torch.from_numpy(stmt_emb), freeze=False)
        self.linear_head_adj_stmt_init = nn.Linear(stmt_embed_size, hidden_size)
        self.linear_tail_adj_stmt_init = nn.Linear(stmt_embed_size, hidden_size)
        self.linear_type_adj_stmt_init = nn.Linear(stmt_embed_size, hidden_size)
        self.linear_head_adj_ere_init = nn.Linear(hidden_size, hidden_size)
        self.linear_tail_adj_ere_init = nn.Linear(hidden_size, hidden_size)
        self.linear_head_adj_stmt = nn.Linear(hidden_size, hidden_size)
        self.linear_tail_adj_stmt = nn.Linear(hidden_size, hidden_size)
        self.linear_type_adj_stmt = nn.Linear(hidden_size, hidden_size)
        self.linear_head_adj_ere = nn.Linear(hidden_size, hidden_size)
        self.linear_tail_adj_ere = nn.Linear(hidden_size, hidden_size)
        self.linear_ere_init = nn.Linear(ere_embed_size, hidden_size)
        self.linear_stmt_init = nn.Linear(stmt_embed_size, hidden_size)
        self.linear_ere = nn.Linear(hidden_size, hidden_size)
        self.linear_stmt = nn.Linear(hidden_size, hidden_size)

        if self_attend:
            self.self_attention_eres_init = Self_Attention(attention_type, hidden_size, attention_size, attention_dropout)
            self.self_attention_stmts_init = Self_Attention(attention_type, hidden_size, attention_size, attention_dropout)
            self.self_attention_eres = Self_Attention(attention_type, hidden_size, attention_size, attention_dropout)
            self.self_attention_stmts = Self_Attention(attention_type, hidden_size, attention_size, attention_dropout)

        self.coherence_attention = Attention(attn_head_stmt_tail, attention_type, hidden_size, attention_size, attention_dropout)
        self.coherence_linear = nn.Linear(attention_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv_dropout = torch.nn.Dropout(p=conv_dropout, inplace=False)
        self.self_attend = self_attend
        self.attention_type = attention_type
        self.attn_head_stmt_tail = attn_head_stmt_tail

        if self.num_layers >= 3:
            self.linear_head_adj_stmt_3 = nn.Linear(hidden_size, hidden_size)
            self.linear_tail_adj_stmt_3 = nn.Linear(hidden_size, hidden_size)
            self.linear_type_adj_stmt_3 = nn.Linear(hidden_size, hidden_size)
            self.linear_head_adj_ere_3 = nn.Linear(hidden_size, hidden_size)
            self.linear_tail_adj_ere_3 = nn.Linear(hidden_size, hidden_size)
            self.linear_ere_3 = nn.Linear(hidden_size, hidden_size)
            self.linear_stmt_3 = nn.Linear(hidden_size, hidden_size)

            torch.nn.init.xavier_uniform_(self.linear_head_adj_stmt_3.weight)
            torch.nn.init.xavier_uniform_(self.linear_tail_adj_stmt_3.weight)
            torch.nn.init.xavier_uniform_(self.linear_type_adj_stmt_3.weight)
            torch.nn.init.xavier_uniform_(self.linear_head_adj_ere_3.weight)
            torch.nn.init.xavier_uniform_(self.linear_tail_adj_ere_3.weight)
            torch.nn.init.xavier_uniform_(self.linear_ere_3.weight)
            torch.nn.init.xavier_uniform_(self.linear_stmt_3.weight)

        if self.num_layers == 4:
            self.linear_head_adj_stmt_4 = nn.Linear(hidden_size, hidden_size)
            self.linear_tail_adj_stmt_4 = nn.Linear(hidden_size, hidden_size)
            self.linear_type_adj_stmt_4 = nn.Linear(hidden_size, hidden_size)
            self.linear_head_adj_ere_4 = nn.Linear(hidden_size, hidden_size)
            self.linear_tail_adj_ere_4 = nn.Linear(hidden_size, hidden_size)
            self.linear_ere_4 = nn.Linear(hidden_size, hidden_size)
            self.linear_stmt_4 = nn.Linear(hidden_size, hidden_size)

            torch.nn.init.xavier_uniform_(self.linear_head_adj_stmt_4.weight)
            torch.nn.init.xavier_uniform_(self.linear_tail_adj_stmt_4.weight)
            torch.nn.init.xavier_uniform_(self.linear_type_adj_stmt_4.weight)
            torch.nn.init.xavier_uniform_(self.linear_head_adj_ere_4.weight)
            torch.nn.init.xavier_uniform_(self.linear_tail_adj_ere_4.weight)
            torch.nn.init.xavier_uniform_(self.linear_ere_4.weight)
            torch.nn.init.xavier_uniform_(self.linear_stmt_4.weight)

        torch.nn.init.xavier_uniform_(self.linear_head_adj_stmt_init.weight)
        torch.nn.init.xavier_uniform_(self.linear_tail_adj_stmt_init.weight)
        torch.nn.init.xavier_uniform_(self.linear_type_adj_stmt_init.weight)
        torch.nn.init.xavier_uniform_(self.linear_head_adj_ere_init.weight)
        torch.nn.init.xavier_uniform_(self.linear_tail_adj_ere_init.weight)

        torch.nn.init.xavier_uniform_(self.linear_head_adj_stmt.weight)
        torch.nn.init.xavier_uniform_(self.linear_tail_adj_stmt.weight)
        torch.nn.init.xavier_uniform_(self.linear_type_adj_stmt.weight)
        torch.nn.init.xavier_uniform_(self.linear_head_adj_ere.weight)
        torch.nn.init.xavier_uniform_(self.linear_tail_adj_ere.weight)

        torch.nn.init.xavier_uniform_(self.linear_ere_init.weight)
        torch.nn.init.xavier_uniform_(self.linear_stmt_init.weight)
        torch.nn.init.xavier_uniform_(self.linear_ere.weight)
        torch.nn.init.xavier_uniform_(self.linear_stmt.weight)
        torch.nn.init.xavier_uniform_(self.coherence_linear.weight)

    def gcn(self, batch, padded_batch_mats, gcn_embeds, device):
        adj_head_list = padded_batch_mats['adj_head']
        adj_tail_list = padded_batch_mats['adj_tail']
        adj_type_list = padded_batch_mats['adj_type']
        ere_labels_list = [batch[iter]['ere_labels'] for iter in range(len(batch))]
        stmt_labels_list = [batch[iter]['stmt_labels'] for iter in range(len(batch))]

        ere_emb = torch.zeros((len(batch), adj_head_list.shape[1], self.ere_embedder.weight.shape[1])).to(device=device)
        stmt_emb = torch.zeros((len(batch), adj_head_list.shape[2], self.stmt_embedder.weight.shape[1])).to(device=device)


        for batch_iter in range(len(batch)):
            for iter in range(len(batch[batch_iter]['ere_labels'])):
                ere_emb[batch_iter][iter] = torch.mean(torch.cat([self.ere_embedder(to_tensor(label_set, device=device)).mean(dim=0).reshape((1, -1)) for label_set in ere_labels_list[batch_iter][iter] if len(label_set) > 0], dim=0), dim=0)
                #ere_emb[iter] = self.ere_embedder(to_tensor([item for sublist in ere_labels[iter] for item in sublist], device=device)).mean(dim=0)

            for iter in range(len(batch[batch_iter]['stmt_labels'])):
                stmt_emb[batch_iter][iter] = torch.mean(torch.cat([self.stmt_embedder(to_tensor(label_set, device=device)).mean(dim=0).reshape((1, -1)) for label_set in stmt_labels_list[batch_iter][iter] if len(label_set) > 0], dim=0), dim=0)
                #stmt_emb[iter] = self.ere_embedder(to_tensor([item for sublist in stmt_labels[iter] for item in sublist], device=device)).mean(dim=0)


        gcn_embeds['eres'] = self.linear_ere_init(ere_emb)
        gcn_embeds['eres'] += torch.bmm(adj_head_list, self.linear_head_adj_stmt_init(stmt_emb))
        gcn_embeds['eres'] += torch.bmm(adj_tail_list, self.linear_tail_adj_stmt_init(stmt_emb))
        gcn_embeds['eres'] += torch.bmm(adj_type_list, self.linear_type_adj_stmt_init(stmt_emb))
        gcn_embeds['eres'] = self.conv_dropout(F.relu(gcn_embeds['eres']))

        gcn_embeds['stmts'] = self.linear_stmt_init(stmt_emb)
        gcn_embeds['stmts'] += torch.bmm(torch.transpose(adj_head_list, 1, 2), self.linear_head_adj_ere_init(gcn_embeds['eres']))
        gcn_embeds['stmts'] += torch.bmm(torch.transpose(adj_tail_list, 1, 2), self.linear_tail_adj_ere_init(gcn_embeds['eres']))
        gcn_embeds['stmts'] = self.conv_dropout(F.relu(gcn_embeds['stmts']))

        gcn_embeds['eres'] = self.linear_ere(gcn_embeds['eres'])
        gcn_embeds['eres'] += torch.bmm(adj_head_list, self.linear_head_adj_stmt(gcn_embeds['stmts']))
        gcn_embeds['eres'] += torch.bmm(adj_tail_list, self.linear_tail_adj_stmt(gcn_embeds['stmts']))
        gcn_embeds['eres'] += torch.bmm(adj_type_list, self.linear_type_adj_stmt(gcn_embeds['stmts']))
        gcn_embeds['eres'] = self.conv_dropout(F.relu(gcn_embeds['eres']))

        gcn_embeds['stmts'] = self.linear_stmt(gcn_embeds['stmts'])
        gcn_embeds['stmts'] += torch.bmm(torch.transpose(adj_head_list, 1, 2), self.linear_head_adj_ere(gcn_embeds['eres']))
        gcn_embeds['stmts'] += torch.bmm(torch.transpose(adj_tail_list, 1, 2), self.linear_tail_adj_ere(gcn_embeds['eres']))
        gcn_embeds['stmts'] = self.conv_dropout(F.relu(gcn_embeds['stmts']))

        if self.num_layers >= 3:
            gcn_embeds['eres'] = self.linear_ere_3(gcn_embeds['eres'])
            gcn_embeds['eres'] += torch.bmm(adj_head_list, self.linear_head_adj_stmt_3(gcn_embeds['stmts']))
            gcn_embeds['eres'] += torch.bmm(adj_tail_list, self.linear_tail_adj_stmt_3(gcn_embeds['stmts']))
            gcn_embeds['eres'] += torch.bmm(adj_type_list, self.linear_type_adj_stmt_3(gcn_embeds['stmts']))
            gcn_embeds['eres'] = self.conv_dropout(F.relu(gcn_embeds['eres']))

            gcn_embeds['stmts'] = self.linear_stmt_3(gcn_embeds['stmts'])
            gcn_embeds['stmts'] += torch.bmm(torch.transpose(adj_head_list, 1, 2), self.linear_head_adj_ere_3(gcn_embeds['eres']))
            gcn_embeds['stmts'] += torch.bmm(torch.transpose(adj_tail_list, 1, 2), self.linear_tail_adj_ere_3(gcn_embeds['eres']))
            gcn_embeds['stmts'] = self.conv_dropout(F.relu(gcn_embeds['stmts']))

        if self.num_layers == 4:
            gcn_embeds['eres'] = self.linear_ere_4(gcn_embeds['eres'])
            gcn_embeds['eres'] += torch.bmm(adj_head_list, self.linear_head_adj_stmt_4(gcn_embeds['stmts']))
            gcn_embeds['eres'] += torch.bmm(adj_tail_list, self.linear_tail_adj_stmt_4(gcn_embeds['stmts']))
            gcn_embeds['eres'] += torch.bmm(adj_type_list, self.linear_type_adj_stmt_4(gcn_embeds['stmts']))
            gcn_embeds['eres'] = self.conv_dropout(F.relu(gcn_embeds['eres']))

            gcn_embeds['stmts'] = self.linear_stmt_4(gcn_embeds['stmts'])
            gcn_embeds['stmts'] += torch.bmm(torch.transpose(adj_head_list, 1, 2), self.linear_head_adj_ere_4(gcn_embeds['eres']))
            gcn_embeds['stmts'] += torch.bmm(torch.transpose(adj_tail_list, 1, 2), self.linear_tail_adj_ere_4(gcn_embeds['eres']))
            gcn_embeds['stmts'] = self.conv_dropout(F.relu(gcn_embeds['stmts']))

    def forward(self, batch, padded_batch_mats, gcn_embeds, device):
        # first gcn layer.
        if not gcn_embeds:
            gcn_embeds = {'eres': dict(), 'stmts': dict()}
            self.gcn(batch, padded_batch_mats, gcn_embeds, device)

        max_query_stmts_size = max([len(batch[iter]['query_stmts']) for iter in range(len(batch))])
        max_query_eres_size = max([len(batch[iter]['query_eres']) for iter in range(len(batch))])
        max_candidates_size = max([len(batch[iter]['candidates']) for iter in range(len(batch))])

        stmt_attendees = torch.zeros((len(batch), max_query_stmts_size, self.hidden_size)).to(device=device)
        ere_attendees = torch.zeros((len(batch), max_query_eres_size, self.hidden_size)).to(device=device)
        attenders = torch.zeros((len(batch), max_candidates_size, self.hidden_size)).to(device=device)

        for iter in range(stmt_attendees.shape[0]):
            stmt_attendees[iter][:len(batch[iter]['query_stmts']), :] = gcn_embeds['stmts'][iter][batch[iter]['query_stmts']]
            ere_attendees[iter][:len(batch[iter]['query_eres']), :] = gcn_embeds['eres'][iter][list(batch[iter]['query_eres'])]
            attenders[iter][:len(batch[iter]['candidates']), :] = gcn_embeds['stmts'][iter][batch[iter]['candidates']]

        mask_stmt_to_stmt = torch.zeros((len(batch), max_query_stmts_size, max_candidates_size), dtype=torch.bool).to(device=device)
        mask_ere_to_stmt = torch.zeros((len(batch), max_query_eres_size, max_candidates_size), dtype=torch.bool).to(device=device)

        for iter in range(mask_stmt_to_stmt.shape[0]):
            mask_stmt_to_stmt[iter][len(batch[iter]['query_stmts']):, :] = 1
            mask_stmt_to_stmt[iter][:, len(batch[iter]['candidates']):] = 1
            mask_ere_to_stmt[iter][len(batch[iter]['query_eres']):, :] = 1
            mask_ere_to_stmt[iter][:, len(batch[iter]['candidates']):] = 1

        coherence_attention_vectors = self.coherence_attention.get_attention_vectors(batch, padded_batch_mats, gcn_embeds, stmt_attendees, ere_attendees, attenders, mask_stmt_to_stmt, mask_ere_to_stmt, device)

        coherence_out = self.coherence_linear(coherence_attention_vectors).squeeze(-1)

        mask = mask_stmt_to_stmt.reshape(-1, mask_stmt_to_stmt.shape[-1])[[i for i in range(0, (len(batch) * max_query_stmts_size), max_query_stmts_size)]]

        coherence_out = torch.Tensor.masked_fill(coherence_out, mask, float('-inf'))

        coherence_prediction = F.log_softmax(coherence_out, dim=1)
        coherence_prediction = torch.Tensor.masked_fill(coherence_prediction, mask, float('-inf'))

        return coherence_prediction, gcn_embeds