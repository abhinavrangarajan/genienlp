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

import os
import math
import numpy as np
import json

import torch
from torch import nn
from torch.nn import functional as F

from ..util import get_trainable_params, set_seed
from ..modules import expectedBLEU, expectedMultiBleu, matrixBLEU

from cove import MTLSTM
from allennlp.modules.elmo import Elmo, batch_to_ids

from .common import *


class MultitaskQuestionAnsweringNetwork(nn.Module):

    def __init__(self, field, args):
        super().__init__()
        self.field = field
        self.args = args
        self.pad_idx = self.field.vocab.stoi[self.field.pad_token]
        self.device = set_seed(args)

        def dp(args):
            return args.dropout_ratio if args.rnn_layers > 1 else 0.

        if self.args.glove_and_char:
        
            self.encoder_embeddings = Embedding(field, args.dimension, 
                dropout=args.dropout_ratio, project=not args.cove)
    
            if self.args.cove or self.args.intermediate_cove:
                self.cove = MTLSTM(model_cache=args.embeddings, layer0=args.intermediate_cove, layer1=args.cove)
                cove_params = get_trainable_params(self.cove) 
                for p in cove_params:
                    p.requires_grad = False
                cove_dim = int(args.intermediate_cove) * 600 + int(args.cove) * 600 + 400 # the last 400 is for GloVe and char n-gram embeddings
                self.project_cove = Feedforward(cove_dim, args.dimension)

        if -1 not in self.args.elmo:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, 3, dropout=0.0, do_layer_norm=False)
            elmo_params = get_trainable_params(self.elmo)
            for p in elmo_params:
                p.requires_grad = False
            elmo_dim = 1024 * len(self.args.elmo)
            self.project_elmo = Feedforward(elmo_dim, args.dimension)
            if self.args.glove_and_char:
                self.project_embeddings = Feedforward(2 * args.dimension, args.dimension, dropout=0.0)
        
        self.decoder_embeddings = Embedding(field, args.dimension, 
            dropout=args.dropout_ratio, project=True)
    
        self.bilstm_before_coattention = PackedLSTM(args.dimension,  args.dimension,
            batch_first=True, bidirectional=True, num_layers=1, dropout=0)
        self.coattention = CoattentiveLayer(args.dimension, dropout=0.3)
        dim = 2*args.dimension + args.dimension + args.dimension

        self.context_bilstm_after_coattention = PackedLSTM(dim, args.dimension,
            batch_first=True, dropout=dp(args), bidirectional=True, 
            num_layers=args.rnn_layers)
        self.self_attentive_encoder_context = TransformerEncoder(args.dimension, args.transformer_heads, args.transformer_hidden, args.transformer_layers, args.dropout_ratio)
        self.bilstm_context = PackedLSTM(args.dimension, args.dimension,
            batch_first=True, dropout=dp(args), bidirectional=True, 
            num_layers=args.rnn_layers)

        self.question_bilstm_after_coattention = PackedLSTM(dim, args.dimension,
            batch_first=True, dropout=dp(args), bidirectional=True, 
            num_layers=args.rnn_layers)
        self.self_attentive_encoder_question = TransformerEncoder(args.dimension, args.transformer_heads, args.transformer_hidden, args.transformer_layers, args.dropout_ratio)
        self.bilstm_question = PackedLSTM(args.dimension, args.dimension,
            batch_first=True, dropout=dp(args), bidirectional=True, 
            num_layers=args.rnn_layers)

        self.self_attentive_decoder = TransformerDecoder(args.dimension, args.transformer_heads, args.transformer_hidden, args.transformer_layers, args.dropout_ratio)
        self.dual_ptr_rnn_decoder = DualPtrRNNDecoder(args.dimension, args.dimension,
            dropout=args.dropout_ratio, num_layers=args.rnn_layers)

        self.generative_vocab_size = min(len(field.vocab), args.max_generative_vocab)
        self.out = nn.Linear(args.dimension, self.generative_vocab_size)

        self.dropout = nn.Dropout(0.4)

    def set_embeddings(self, embeddings):
        self.encoder_embeddings.set_embeddings(embeddings)
        self.decoder_embeddings.set_embeddings(embeddings)

    def forward(self, batch, iteration):
        context, context_lengths, context_limited, context_elmo    = batch.context,  batch.context_lengths,  batch.context_limited, batch.context_elmo
        question, question_lengths, question_limited, question_elmo = batch.question, batch.question_lengths, batch.question_limited, batch.question_elmo
        answer, answer_lengths, answer_limited       = batch.answer,   batch.answer_lengths,   batch.answer_limited
        oov_to_limited_idx, limited_idx_to_full_idx  = batch.oov_to_limited_idx, batch.limited_idx_to_full_idx

        def map_to_full(x):
            return limited_idx_to_full_idx[x]
        self.map_to_full = map_to_full

        if -1 not in self.args.elmo:
            def elmo(z, layers, device):
                e = self.elmo(batch_to_ids(z).to(device))['elmo_representations']
                return torch.cat([e[x] for x in layers], -1)
            context_elmo =  self.project_elmo(elmo(context_elmo, self.args.elmo, context.device).detach())
            question_elmo = self.project_elmo(elmo(question_elmo, self.args.elmo, question.device).detach())

        if self.args.glove_and_char:
            context_embedded = self.encoder_embeddings(context)
            question_embedded = self.encoder_embeddings(question)
            if self.args.cove:
                context_embedded = self.project_cove(torch.cat([self.cove(context_embedded[:, :, -300:], context_lengths), context_embedded], -1).detach())
                question_embedded = self.project_cove(torch.cat([self.cove(question_embedded[:, :, -300:], question_lengths), question_embedded], -1).detach())
            if -1 not in self.args.elmo:
                context_embedded = self.project_embeddings(torch.cat([context_embedded, context_elmo], -1))
                question_embedded = self.project_embeddings(torch.cat([question_embedded, question_elmo], -1))
        else:
            context_embedded, question_embedded = context_elmo, question_elmo 

        context_encoded = self.bilstm_before_coattention(context_embedded, context_lengths)[0]
        question_encoded = self.bilstm_before_coattention(question_embedded, question_lengths)[0]

        context_padding = context.data == self.pad_idx
        question_padding = question.data == self.pad_idx

        coattended_context, coattended_question = self.coattention(context_encoded, question_encoded, context_padding, question_padding)

        context_summary = torch.cat([coattended_context, context_encoded, context_embedded], -1)
        condensed_context, _ = self.context_bilstm_after_coattention(context_summary, context_lengths)
        self_attended_context = self.self_attentive_encoder_context(condensed_context, padding=context_padding)
        final_context, (context_rnn_h, context_rnn_c) = self.bilstm_context(self_attended_context[-1], context_lengths)
        context_rnn_state = [reshape_rnn_state(x) for x in (context_rnn_h, context_rnn_c)]

        question_summary = torch.cat([coattended_question, question_encoded, question_embedded], -1)
        condensed_question, _ = self.question_bilstm_after_coattention(question_summary, question_lengths)
        self_attended_question = self.self_attentive_encoder_question(condensed_question, padding=question_padding)
        final_question, (question_rnn_h, question_rnn_c) = self.bilstm_question(self_attended_question[-1], question_lengths)
        question_rnn_state = [reshape_rnn_state(x) for x in (question_rnn_h, question_rnn_c)]

        context_indices = context_limited if context_limited is not None else context
        question_indices = question_limited if question_limited is not None else question
        answer_indices = answer_limited if answer_limited is not None else answer

        pad_idx = self.field.decoder_stoi[self.field.pad_token]
        context_padding = context_indices.data == pad_idx
        question_padding = question_indices.data == pad_idx

        self.dual_ptr_rnn_decoder.applyMasks(context_padding, question_padding)

        if self.training:
            answer_padding = (answer_indices.data == pad_idx)[:, :-1]
            answer_embedded = self.decoder_embeddings(answer)
            self_attended_decoded = self.self_attentive_decoder(answer_embedded[:, :-1].contiguous(), self_attended_context, context_padding=context_padding, answer_padding=answer_padding, positional_encodings=True)
            decoder_outputs = self.dual_ptr_rnn_decoder(self_attended_decoded, 
                final_context, final_question, hidden=context_rnn_state)
            rnn_output, context_attention, question_attention, context_alignment, question_alignment, vocab_pointer_switch, context_question_switch, rnn_state = decoder_outputs

            probs = self.probs(self.out, rnn_output, vocab_pointer_switch, context_question_switch, 
                context_attention, question_attention, 
                context_indices, question_indices, 
                oov_to_limited_idx)


            if self.args.use_bleu_loss and iteration >= self.args.loss_switch * max(self.args.train_iterations):
                max_order = 4
                targets = answer_indices[:, 1:].contiguous()
                batch_size = targets.size(0)
                reference_lengths = [l-1 for l in answer_lengths]
                translation_len = max(reference_lengths)
                translation_lengths = torch.tensor([translation_len] * batch_size, device=self.device)

                bleu_loss_smoothed = expectedMultiBleu.bleu(probs, targets, translation_lengths, reference_lengths, max_order=max_order, smooth=True)
                loss = -1 * bleu_loss_smoothed[0]

            elif self.args.use_maxmargin_loss:
                targets = answer_indices[:, 1:].contiguous()
                loss = max_margin_loss(probs, targets, pad_idx=pad_idx)

            else:
                probs, targets = mask(answer_indices[:, 1:].contiguous(), probs.contiguous(), pad_idx=pad_idx)
                loss = F.nll_loss(probs.log(), targets)
            return loss, None

        else:

            if self.args.beam_search:
                return None, self.beam_search(self_attended_context, final_context, final_question,
                context_indices, question_indices,
                oov_to_limited_idx, rnn_state=context_rnn_state).data

            else:
                return None, self.greedy(self_attended_context, final_context, final_question,
                    context_indices, question_indices,
                    oov_to_limited_idx, rnn_state=context_rnn_state).data


    def probs(self, generator, outputs, vocab_pointer_switches, context_question_switches, 
        context_attention, question_attention, 
        context_indices, question_indices, 
        oov_to_limited_idx):

        size = list(outputs.size())

        size[-1] = self.generative_vocab_size
        scores = generator(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim()-1)
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = self.generative_vocab_size + len(oov_to_limited_idx)
        if self.generative_vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = scaled_p_vocab.new_full(size, EPSILON)
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim()-1)

        # p_context_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim()-1, context_indices.unsqueeze(1).expand_as(context_attention), 
            (context_question_switches * (1 - vocab_pointer_switches)).expand_as(context_attention) * context_attention)

        # p_question_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim()-1, question_indices.unsqueeze(1).expand_as(question_attention), 
            ((1 - context_question_switches) * (1 - vocab_pointer_switches)).expand_as(question_attention) * question_attention)

        return scaled_p_vocab



    def beam_search(self, self_attended_context, context, question, context_indices, question_indices, oov_to_limited_idx, rnn_state=None):

        K = self.args.K
        B, TC, C = context.size()
        T = self.args.max_output_length

        vocab_size = self.generative_vocab_size + len(oov_to_limited_idx)

        pos_index = (torch.LongTensor(range(B)) * K).view(-1, 1)

        num_layers = len(self.self_attentive_decoder.layers)

        self_attended_context = [inflate(tensor, K, 0) for tensor in self_attended_context]
        rnn_state = [inflate(state, K, 1) for state in rnn_state]
        context = inflate(context, K, 0)
        question = inflate(question, K, 0)
        context_indices = inflate(context_indices, K, 0)
        question_indices = inflate(question_indices, K, 0)

        ################
        pad_idx = self.field.decoder_stoi[self.field.pad_token]
        context_padding = context_indices.data == pad_idx
        question_padding = question_indices.data == pad_idx

        self.dual_ptr_rnn_decoder.applyMasks(context_padding, question_padding)
        ################

        hiddens = [self_attended_context[0].new_zeros((B*K, T, C))
           for l in range(num_layers+1)]

        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])

        sequence_scores = torch.Tensor(B * K, 1)
        sequence_scores.fill_(-float('inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * K for i in range(0, B)]), 0.0)


        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        rnn_output, context_alignment, question_alignment = None, None, None
        input = self_attended_context[-1].new_full((B*K, 1), self.field.vocab.stoi['<init>'], dtype=torch.long)

        for t in range(T):
            input_var = self.decoder_embeddings(input, [1]*B*K)

            hiddens[0][:, t] = hiddens[0][:, t] + (math.sqrt(self.self_attentive_decoder.d_model) * input_var).squeeze(1)

            for l in range(num_layers):
                hiddens[l + 1][:, t] = self.self_attentive_decoder.layers[l].feedforward(
                    self.self_attentive_decoder.layers[l].attention(
                    self.self_attentive_decoder.layers[l].selfattn(hiddens[l][:, t], hiddens[l][:, :t + 1], hiddens[l][:, :t + 1])
                  , self_attended_context[l], self_attended_context[l]))

            decoder_outputs = self.dual_ptr_rnn_decoder(hiddens[-1][:, t].unsqueeze(1),
                context, question,
                context_alignment=context_alignment, question_alignment=question_alignment,
                hidden=rnn_state, output=rnn_output)

            rnn_output, context_attention, question_attention, context_alignment, question_alignment, vocab_pointer_switch, context_question_switch, rnn_state = decoder_outputs
            probs = self.probs(self.out, rnn_output, vocab_pointer_switch, context_question_switch,
                context_attention, question_attention,
                context_indices, question_indices,
                oov_to_limited_idx)

            log_softmax_output = torch.log(probs)
            stored_outputs.append(probs.squeeze(1))

            sequence_scores = inflate(sequence_scores, vocab_size, 1)
            sequence_scores += log_softmax_output.squeeze(1)
            scores, candidates = sequence_scores.view(B, -1).topk(K, dim=1)

            input = (candidates % vocab_size).view(B*K, 1)
            sequence_scores = scores.view(B*K, 1)

            # update params for next step
            predecessors = (candidates / vocab_size + pos_index.expand_as(candidates)).view(B*K, 1)
            for l in range(num_layers+1):
                hiddens[l][:, t] = hiddens[l][:, t].index_select(0, predecessors.squeeze(1))

            hiddens_for_t = [hiddens[l][:, t] for l in range(num_layers+1)]
            concat_hiddens = torch.stack(hiddens_for_t, dim=0)  # (layer, b*k, hidden)


            # update stored values
            stored_scores.append(sequence_scores.clone())

            eos_indices = input.data.eq(self.field.vocab.stoi['<eos>'])
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input)
            stored_hidden.append(concat_hiddens)

        # assert sizes are correct
        assert len(stored_outputs) == len(stored_hidden) == len(stored_predecessors) == len(stored_emitted_symbols) == len(stored_scores) == T
        assert list(stored_outputs[0].size()) == [B*K, vocab_size]
        assert list(stored_hidden[0].size()) == [num_layers+1, B*K, C]
        assert list(stored_predecessors[0].size()) == [B*K, 1]
        assert list(stored_emitted_symbols[0].size()) == [B*K, 1]
        assert list(stored_scores[0].size()) == [B*K, 1]


        # do backtracking
        output, h_t, h_n, score, length, p = backtrack(stored_outputs, stored_hidden, stored_predecessors, stored_emitted_symbols, stored_scores, B, C, K, T, self.field.vocab.stoi['<eos>'])
        p = torch.cat(p, dim=-1)
        length = torch.tensor(length)

        # Assert output sizes are correct
        assert len(output) == len(h_t) == T
        assert list(output[0].size()) == [B, K, vocab_size]
        assert list(h_t[0].size()) == [num_layers+1, B, K, C]
        assert list(h_n.size()) == [num_layers+1, B, K, C]
        assert list(score.size()) == list(length.size()) == [B, K]
        assert list(p.size()) == [B, K, T]


        # choose sequence with highest score

        outs = context.new_full((B, T), self.field.decoder_stoi['<pad>'], dtype=torch.long)

        sequences = torch.stack([seq[0] for seq in p], dim=0)
        lengths = torch.tensor([seq_len[0] for seq_len in length])

        assert list(sequences.size()) == [B, T]
        assert list(lengths.size()) == [B]

        def generate_mask(length, max_len):
            assert length.dim() == 1
            mask = torch.arange(max_len).expand(length.size(0), max_len) < length.unsqueeze(1)
            return mask

        mask = generate_mask(lengths, T)
        outs = torch.where(mask, sequences, outs)

        assert list(outs.size()) == [B, T]

        return outs


    def greedy(self, self_attended_context, context, question, context_indices, question_indices, oov_to_limited_idx, rnn_state=None):
        B, TC, C = context.size()
        T = self.args.max_output_length
        outs = context.new_full((B, T), self.field.decoder_stoi['<pad>'], dtype=torch.long)
        hiddens = [self_attended_context[0].new_zeros((B, T, C))
                   for l in range(len(self.self_attentive_decoder.layers) + 1)]
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = context.new_zeros((B, )).byte()
    
        rnn_output, context_alignment, question_alignment = None, None, None
        for t in range(T):
            if t == 0:
                embedding = self.decoder_embeddings(
                    self_attended_context[-1].new_full((B, 1), self.field.vocab.stoi['<init>'], dtype=torch.long), [1]*B)
            else:
                embedding = self.decoder_embeddings(outs[:, t - 1].unsqueeze(1), [1]*B)
            hiddens[0][:, t] = hiddens[0][:, t] + (math.sqrt(self.self_attentive_decoder.d_model) * embedding).squeeze(1)
            for l in range(len(self.self_attentive_decoder.layers)):
                hiddens[l + 1][:, t] = self.self_attentive_decoder.layers[l].feedforward(
                    self.self_attentive_decoder.layers[l].attention(
                    self.self_attentive_decoder.layers[l].selfattn(hiddens[l][:, t], hiddens[l][:, :t + 1], hiddens[l][:, :t + 1])
                  , self_attended_context[l], self_attended_context[l]))
            decoder_outputs = self.dual_ptr_rnn_decoder(hiddens[-1][:, t].unsqueeze(1),
                context, question, 
                context_alignment=context_alignment, question_alignment=question_alignment,
                hidden=rnn_state, output=rnn_output)
            rnn_output, context_attention, question_attention, context_alignment, question_alignment, vocab_pointer_switch, context_question_switch, rnn_state = decoder_outputs
            probs = self.probs(self.out, rnn_output, vocab_pointer_switch, context_question_switch, 
                context_attention, question_attention, 
                context_indices, question_indices, 
                oov_to_limited_idx)
            pred_probs, preds = probs.max(-1)
            preds = preds.squeeze(1)
            eos_yet = eos_yet | (preds == self.field.decoder_stoi['<eos>'])
            outs[:, t] = preds.cpu().apply_(self.map_to_full)
            if eos_yet.all():
                break
        return outs





class DualPtrRNNDecoder(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.0, num_layers=1):
        super().__init__()
        self.d_hid = d_hid
        self.d_in = d_in
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.input_feed = True
        if self.input_feed:
            d_in += 1 * d_hid

        self.rnn = LSTMDecoder(self.num_layers, d_in, d_hid, dropout)
        self.context_attn = LSTMDecoderAttention(d_hid, dot=True)
        self.question_attn = LSTMDecoderAttention(d_hid, dot=True)

        self.vocab_pointer_switch = nn.Sequential(Feedforward(2 * self.d_hid + d_in, 1), nn.Sigmoid())
        self.context_question_switch = nn.Sequential(Feedforward(2 * self.d_hid + d_in, 1), nn.Sigmoid())

    def forward(self, input, context, question, output=None, hidden=None, context_alignment=None, question_alignment=None):
        context_output = output.squeeze(1) if output is not None else self.make_init_output(context)
        context_alignment = context_alignment if context_alignment is not None else self.make_init_output(context)
        question_alignment = question_alignment if question_alignment is not None else self.make_init_output(question)

        context_outputs, vocab_pointer_switches, context_question_switches, context_attentions, question_attentions, context_alignments, question_alignments = [], [], [], [], [], [], []
        for emb_t in input.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            context_output = self.dropout(context_output)
            if self.input_feed:
                emb_t = torch.cat([emb_t, context_output], 1)
            dec_state, hidden = self.rnn(emb_t, hidden)
            context_output, context_attention, context_alignment = self.context_attn(dec_state, context)
            question_output, question_attention, question_alignment = self.question_attn(dec_state, question)
            vocab_pointer_switch = self.vocab_pointer_switch(torch.cat([dec_state, context_output, emb_t], -1))
            context_question_switch = self.context_question_switch(torch.cat([dec_state, question_output, emb_t], -1))
            context_output = self.dropout(context_output)
            context_outputs.append(context_output)
            vocab_pointer_switches.append(vocab_pointer_switch)
            context_question_switches.append(context_question_switch)
            context_attentions.append(context_attention)
            context_alignments.append(context_alignment)
            question_attentions.append(question_attention)
            question_alignments.append(question_alignment)

        context_outputs, vocab_pointer_switches, context_question_switches, context_attention, question_attention = [self.package_outputs(x) for x in [context_outputs, vocab_pointer_switches, context_question_switches, context_attentions, question_attentions]]
        return context_outputs, context_attention, question_attention, context_alignment, question_alignment, vocab_pointer_switches, context_question_switches, hidden


    def applyMasks(self, context_mask, question_mask):
        self.context_attn.applyMasks(context_mask)
        self.question_attn.applyMasks(question_mask)

    def make_init_output(self, context):
        batch_size = context.size(0)
        h_size = (batch_size, self.d_hid)
        return context.new_zeros(h_size)

    def package_outputs(self, outputs):
        outputs = torch.stack(outputs, dim=1)
        return outputs
