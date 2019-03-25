#
# Copyright (c) 2018, Salesforce, Inc.
#                     The Board of Trustees of the Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import os
import sys
import numpy as np
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


INF = 1e10
EPSILON = 1e-10

class LSTMDecoder(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(LSTMDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            input = self.dropout(input)
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


# adopted from https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py
# pytorch seq2seq framework by IBM
def backtrack(nw_output, nw_hidden, predecessors, symbols, scores, batch_size, hidden_size, K, T, eos_id):
    """Backtracks over batch to generate optimal k-sequences.
        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            batch_size: Size of the batch
            hidden_size: Size of the hidden state
        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.
            score [batch, k]: A list containing the final scores for all top-k sequences
            length [batch, k]: A list specifying the length of each sequence in the top-k candidates
            p (batch, k, sequence_len): A Tensor containing predicted sequence
        """


    output = list()
    h_t = list()
    p = list()

    h_n = torch.zeros(nw_hidden[0].size())
    l = [[T] * K for _ in range(batch_size)]  # Placeholder for lengths of top-k sequences # Similar to `h_n`

    # the last step output of the beams are not sorted
    # thus they are sorted here
    sorted_score, sorted_idx = scores[-1].view(batch_size, K).topk(K)
    # initialize the sequence scores with the sorted last step beam scores
    s = sorted_score.clone()

    batch_eos_found = [0] * batch_size   # the number of EOS found in the backward loop below for each batch
    pos_index = (torch.LongTensor(range(batch_size)) * K).view(-1, 1)
    t = T - 1
    # initialize the back pointer with the sorted order of the last step beams.
    # add pos_index for indexing variable with b*k as the first dimension.
    t_predecessors = (sorted_idx + pos_index.expand_as(sorted_idx)).view(batch_size * K)
    while t >= 0:
        # Re-order the variables with the back pointer
        current_output = nw_output[t].index_select(0, t_predecessors)

        current_hidden = nw_hidden[t].index_select(1, t_predecessors)
        current_symbol = symbols[t].index_select(0, t_predecessors)
        # Re-order the back pointer of the previous step with the back pointer of
        # the current step
        t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()


        eos_indices = symbols[t].data.squeeze(1).eq(eos_id).nonzero()
        if eos_indices.dim() > 0:
            for i in range(eos_indices.size(0)-1, -1, -1):
                # Indices of the EOS symbol for both variables
                # with b*k as the first dimension, and b, k for
                # the first two dimensions
                idx = eos_indices[i]
                b_idx = int(idx[0] / K)
                # The indices of the replacing position
                # according to the replacement strategy noted above
                res_k_idx = K - (batch_eos_found[b_idx] % K) - 1
                batch_eos_found[b_idx] += 1
                res_idx = b_idx * K + res_k_idx

                # Replace the old information in return variables
                # with the new ended sequence information
                t_predecessors[res_idx] = predecessors[t][idx[0]]
                current_output[res_idx, :] = nw_output[t][idx[0], :]

                current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]
                h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data
                current_symbol[res_idx, :] = symbols[t][idx[0]]
                s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                l[b_idx][res_k_idx] = t + 1

        # record the back tracked results
        output.append(current_output)
        h_t.append(current_hidden)
        p.append(current_symbol)

        t -= 1

    # Sort and re-order again as the added ended sequences may change
    # the order (very unlikely)
    s, re_sorted_idx = s.topk(K)
    for b_idx in range(batch_size):
        l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

    re_sorted_idx = (re_sorted_idx + pos_index.expand_as(re_sorted_idx)).view(batch_size * K)

    # Reverse the sequences and re-order at the same time
    # It is reversed because the backtracking happens in reverse time order
    output = [step.index_select(0, re_sorted_idx).view(batch_size, K, -1) for step in reversed(output)]
    p = [step.index_select(0, re_sorted_idx).view(batch_size, K, -1) for step in reversed(p)]
    h_t = [step.index_select(1, re_sorted_idx).view(-1, batch_size, K, hidden_size) for step in reversed(h_t)]
    h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, batch_size, K, hidden_size)
    s = s.data

    return output, h_t, h_n, s, l, p

def reshape_rnn_state(h):
    return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
            .transpose(1, 2).contiguous() \
            .view(h.size(0) // 2, h.size(1), h.size(2) * 2).contiguous()

def inflate(tensor, rep, dim):
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = rep
    return tensor.repeat(*repeat_dims)

def max_margin_loss(probs, targets, pad_idx=1):

    batch_size, max_length, depth = probs.size()
    targets_mask = (targets != pad_idx).float()
    flat_mask = targets_mask.view(batch_size*max_length,)
    flat_preds = probs.view(batch_size*max_length, depth)

    one_hot = torch.zeros_like(probs)
    one_hot_gold = one_hot.scatter_(2, targets.unsqueeze(2), 1)

    marginal_scores = probs - one_hot_gold + 1
    marginal_scores = marginal_scores.view(batch_size*max_length, depth)
    max_margin = torch.max(marginal_scores, dim=1)[0]

    gold_score = torch.masked_select(flat_preds, one_hot_gold.view(batch_size*max_length, depth).byte())
    margin = max_margin - gold_score

    return torch.sum(margin*flat_mask) + 1e-8


def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0., x.size(1))
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)


# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)


def pad_to_match(x, y):
    x_len, y_len = x.size(1), y.size(1)
    if x_len == y_len:
        return x, y
    extra = x.new_ones((x.size(0), abs(y_len - x_len)))
    if x_len < y_len:
        return torch.cat((x, extra), 1), y
    return x, torch.cat((y, extra), 1)


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, dropout_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x, padding=None):
        return self.layernorm(x[0] + self.dropout(self.layer(*x, padding=padding)))


class Attention(nn.Module):

    def __init__(self, d_key, dropout_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(dropout_ratio)
        self.causal = causal

    def forward(self, query, key, value, padding=None):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and self.causal:
            tri = key.new_ones((key.size(1), key.size(1))).triu(1) * INF
            dot_products.sub_(tri.unsqueeze(0))
        if not padding is None:
            dot_products.masked_fill_(padding.unsqueeze(1).expand_as(dot_products), -INF)
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)


class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, dropout_ratio, causal=False):
        super().__init__()
        self.attention = Attention(d_key, dropout_ratio, causal=causal)
        self.wq = Linear(d_key, d_key, bias=False)
        self.wk = Linear(d_key, d_key, bias=False)
        self.wv = Linear(d_value, d_value, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value, padding=None):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return torch.cat([self.attention(q, k, v, padding=padding)
                          for q, k, v in zip(query, key, value)], -1)


class LinearReLU(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.feedforward = Feedforward(d_model, d_hidden, activation='relu')
        self.linear = Linear(d_hidden, d_model)

    def forward(self, x, padding=None):
        return self.linear(self.feedforward(x))


class TransformerEncoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(
                dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, x, padding=None):
        return self.feedforward(self.selfattn(x, x, x, padding=padding))


class TransformerEncoder(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dimension, n_heads, hidden, dropout) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding=None):
        x = self.dropout(x)
        encoding = [x]
        for layer in self.layers:
            x = layer(x, padding=padding)
            encoding.append(x)
        return encoding


class TransformerDecoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout, causal=True):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout, causal),
            dimension, dropout)
        self.attention = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, x, encoding, context_padding=None, answer_padding=None):
        x = self.selfattn(x, x, x, padding=answer_padding)
        return self.feedforward(self.attention(x, encoding, encoding, padding=context_padding))


class TransformerDecoder(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout, causal=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(dimension, n_heads, hidden, dropout, causal=causal) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = dimension

    def forward(self, x, encoding, context_padding=None, positional_encodings=True, answer_padding=None):
        if positional_encodings:
            x = x + positional_encodings_like(x)
        x = self.dropout(x)
        for layer, enc in zip(self.layers, encoding[1:]):
            x = layer(x, enc, context_padding=context_padding, answer_padding=answer_padding)
        return x


def mask(targets, out, squash=True, pad_idx=1):
    mask = (targets != pad_idx)
    out_mask = mask.unsqueeze(-1).expand_as(out).contiguous()
    if squash:
        out_after = out[out_mask].contiguous().view(-1, out.size(-1))
    else:
        out_after = out * out_mask.float()
    targets_after = targets[mask]
    return out_after, targets_after





class Highway(torch.nn.Module):
    def __init__(self, d_in, activation='relu', n_layers=1):
        super(Highway, self).__init__()
        self.d_in = d_in
        self._layers = torch.nn.ModuleList([Linear(d_in, 2 * d_in) for _ in range(n_layers)])
        for layer in self._layers:
            layer.bias[d_in:].fill_(1)
        self.activation = getattr(F, activation)

    def forward(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, :self.d_in] if projected_input.dim() == 2 else projected_input[:, :, :self.d_in]
            nonlinear_part = self.activation(nonlinear_part)
            gate = projected_input[:, self.d_in:(2 * self.d_in)] if projected_input.dim() == 2 else projected_input[:, :, self.d_in:(2 * self.d_in)]
            gate = F.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class LinearFeedforward(nn.Module):

    def __init__(self, d_in, d_hid, d_out, activation='relu'):
        super().__init__()
        self.feedforward = Feedforward(d_in, d_hid, activation=activation)
        self.linear = Linear(d_hid, d_out)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.linear(self.feedforward(x)))

class PackedLSTM(nn.Module):

    def __init__(self, d_in, d_out, bidirectional=False, num_layers=1, 
        dropout=0.0, batch_first=True):
        """A wrapper class that packs input sequences and unpacks output sequences"""
        super().__init__()
        if bidirectional:
            d_out = d_out // 2
        self.rnn = nn.LSTM(d_in, d_out,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.batch_first = batch_first

    def forward(self, inputs, lengths, hidden=None):
        lens, indices = torch.sort(inputs.new_tensor(lengths, dtype=torch.long), 0, True)
        inputs = inputs[indices] if self.batch_first else inputs[:, indices] 
        outputs, (h, c) = self.rnn(pack(inputs, lens.tolist(), 
            batch_first=self.batch_first), hidden)
        outputs = unpack(outputs, batch_first=self.batch_first)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:, _indices]
        h, c = h[:, _indices, :], c[:, _indices, :]
        return outputs, (h, c)


class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


class Feedforward(nn.Module):

    def __init__(self, d_in, d_out, activation=None, bias=True, dropout=0.2):
        super().__init__()
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = lambda x: x
        self.linear = Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.activation(self.linear(self.dropout(x)))


class Embedding(nn.Module):

    def __init__(self, field, output_dimension, include_pretrained=True, trained_dimension=0, dropout=0.0, project=True):
        super().__init__()
        self.field = field
        self.project = project
        dimension = 0
        pretrained_dimension = field.vocab.vectors.size(-1)

        if include_pretrained:
            # NOTE: this must be a list so that pytorch will not iterate into the module when
            # traversing this module
            # in turn, this means that moving this Embedding() to the GPU will not move the
            # actual embedding, which will stay on CPU; this is necessary because a) we call
            # set_embeddings() sometimes with CPU-only tensors, and b) the embedding tensor
            # is too big for the GPU anyway
            self.pretrained_embeddings = [nn.Embedding(len(field.vocab), pretrained_dimension)]
            self.pretrained_embeddings[0].weight.data = field.vocab.vectors
            self.pretrained_embeddings[0].weight.requires_grad = False
            dimension += pretrained_dimension
        else:
            self.pretrained_embeddings = None

        # OTOH, if we have a trained embedding, we move it around together with the module
        # (ie, potentially on GPU), because the saving when applying gradient outweights
        # the cost, and hopefully the embedding is small enough to fit in GPU memory
        if trained_dimension > 0:
            self.trained_embeddings = nn.Embedding(len(field.vocab), trained_dimension)
            dimension += trained_dimension
        else:
            self.trained_embeddings = None
        if self.project:
            self.projection = Feedforward(dimension, output_dimension)
        self.dropout = nn.Dropout(dropout)
        self.dimension = output_dimension

    def forward(self, x, lengths=None, device=-1):
        if self.pretrained_embeddings is not None:
            pretrained_embeddings = self.pretrained_embeddings[0](x.cpu()).to(x.device).detach()
        else:
            pretrained_embeddings = None
        if self.trained_embeddings is not None:
            trained_vocabulary_size = self.trained_embeddings.weight.size()[0]
            valid_x = torch.lt(x, trained_vocabulary_size)
            masked_x = torch.where(valid_x, x, torch.zeros_like(x))
            trained_embeddings = self.trained_embeddings(masked_x)
        else:
            trained_embeddings = None
        if pretrained_embeddings is not None and trained_embeddings is not None:
            embeddings = torch.cat((pretrained_embeddings, trained_embeddings), dim=2)
        elif pretrained_embeddings is not None:
            embeddings = pretrained_embeddings
        else:
            embeddings = trained_embeddings

        return self.projection(embeddings) if self.project else embeddings

    def set_embeddings(self, w):
        if self.pretrained_embeddings is not None:
            self.pretrained_embeddings[0].weight.data = w
            self.pretrained_embeddings[0].weight.requires_grad = False


class SemanticFusionUnit(nn.Module):
 
    def __init__(self, d, l):
        super().__init__()
        self.r_hat = Feedforward(d*l, d, 'tanh')
        self.g = Feedforward(d*l, d, 'sigmoid')
        self.dropout = nn.Dropout(0.2)
 
    def forward(self, x):
        c = self.dropout(torch.cat(x, -1))
        r_hat = self.r_hat(c)
        g = self.g(c)
        o = g * r_hat + (1 - g) * x[0]
        return o


class LSTMDecoderAttention(nn.Module):
    def __init__(self, dim, dot=False):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(2 * dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.dot = dot

    def applyMasks(self, context_mask):
        self.context_mask = context_mask

    def forward(self, input, context):
        if not self.dot:
            targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
        else:
            targetT = input.unsqueeze(2)

        context_scores = torch.bmm(context, targetT).squeeze(2)
        context_scores.masked_fill_(self.context_mask, -float('inf'))
        context_attention = F.softmax(context_scores, dim=-1) + EPSILON
        context_alignment = torch.bmm(context_attention.unsqueeze(1), context).squeeze(1)

        combined_representation = torch.cat([input, context_alignment], 1)
        output = self.tanh(self.linear_out(combined_representation))

        return output, context_attention, context_alignment


class CoattentiveLayer(nn.Module):

    def __init__(self, d, dropout=0.2):
        super().__init__()
        self.proj = Feedforward(d, d, dropout=0.0)
        self.embed_sentinel = nn.Embedding(2, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, question, context_padding, question_padding): 
        context_padding = torch.cat([context.new_zeros((context.size(0), 1), dtype=torch.long)==1, context_padding], 1)
        question_padding = torch.cat([question.new_zeros((question.size(0), 1), dtype=torch.long)==1, question_padding], 1)

        context_sentinel = self.embed_sentinel(context.new_zeros((context.size(0), 1), dtype=torch.long))
        context = torch.cat([context_sentinel, self.dropout(context)], 1) # batch_size x (context_length + 1) x features

        question_sentinel = self.embed_sentinel(question.new_ones((question.size(0), 1), dtype=torch.long))
        question = torch.cat([question_sentinel, question], 1) # batch_size x (question_length + 1) x features
        question = torch.tanh(self.proj(question)) # batch_size x (question_length + 1) x features

        affinity = context.bmm(question.transpose(1,2)) # batch_size x (context_length + 1) x (question_length + 1)
        attn_over_context = self.normalize(affinity, context_padding) # batch_size x (context_length + 1) x 1
        attn_over_question = self.normalize(affinity.transpose(1,2), question_padding) # batch_size x (question_length + 1) x 1
        sum_of_context = self.attn(attn_over_context, context) # batch_size x (question_length + 1) x features
        sum_of_question = self.attn(attn_over_question, question) # batch_size x (context_length + 1) x features
        coattn_context = self.attn(attn_over_question, sum_of_context) # batch_size x (context_length + 1) x features
        coattn_question = self.attn(attn_over_context, sum_of_question) # batch_size x (question_length + 1) x features
        return torch.cat([coattn_context, sum_of_question], 2)[:, 1:], torch.cat([coattn_question, sum_of_context], 2)[:, 1:]

    @staticmethod
    def attn(weights, candidates):
        w1, w2, w3 = weights.size()
        c1, c2, c3 = candidates.size()
        return weights.unsqueeze(3).expand(w1, w2, w3, c3).mul(candidates.unsqueeze(2).expand(c1, c2, w3, c3)).sum(1).squeeze(1)

    @staticmethod
    def normalize(original, padding):
        raw_scores = original.clone()
        raw_scores.masked_fill_(padding.unsqueeze(-1).expand_as(raw_scores), -INF)
        return F.softmax(raw_scores, dim=1)

