import os
import math
import numpy as np
import json
import torch
from torch import nn
from torch.nn import functional as F

from util import get_trainable_params, set_seed

from .common import positional_encodings_like, INF, EPSILON, TransformerEncoder, TransformerDecoder, PackedLSTM, LSTMDecoderAttention, LSTMDecoder, Embedding, Feedforward, mask, CoattentiveLayer


class ClassifierNetwork(nn.Module):

    def __init__(self, field, args):
        super().__init__()
        self.field = field
        self.args = args
        self.pad_idx = self.field.vocab.stoi[self.field.pad_token]
        self.device = set_seed(args)

        def dp(args):
            return args.dropout_ratio if args.rnn_layers > 1 else 0.

        self.encoder_embeddings = Embedding(field, args.dimension,
                                            dropout=args.dropout_ratio, project=not (args.cove or args.elmo))
        self.decoder_embeddings = Embedding(field, args.dimension,
                                            dropout=args.dropout_ratio, project=True)


        self.bilstm_before_coattention = PackedLSTM(args.dimension, args.dimension,
                                                    batch_first=True, bidirectional=True, num_layers=1, dropout=0)
        self.coattention = CoattentiveLayer(args.dimension, dropout=0.3)
        dim = 2 * args.dimension + args.dimension + args.dimension

        self.context_bilstm_after_coattention = PackedLSTM(dim, args.dimension,
                                                           batch_first=True, dropout=dp(args), bidirectional=True,
                                                           num_layers=args.rnn_layers)
        self.self_attentive_encoder_context = TransformerEncoder(args.dimension, args.transformer_heads,
                                                                 args.transformer_hidden, args.transformer_layers,
                                                                 args.dropout_ratio)
        self.bilstm_context = PackedLSTM(args.dimension, args.dimension,
                                         batch_first=True, dropout=dp(args), bidirectional=True,
                                         num_layers=args.rnn_layers)

        self.question_bilstm_after_coattention = PackedLSTM(dim, args.dimension,
                                                            batch_first=True, dropout=dp(args), bidirectional=True,
                                                            num_layers=args.rnn_layers)
        self.self_attentive_encoder_question = TransformerEncoder(args.dimension, args.transformer_heads,
                                                                  args.transformer_hidden, args.transformer_layers,
                                                                  args.dropout_ratio)
        self.bilstm_question = PackedLSTM(args.dimension, args.dimension,
                                          batch_first=True, dropout=dp(args), bidirectional=True,
                                          num_layers=args.rnn_layers)

        self.self_attentive_decoder = TransformerDecoder(args.dimension, args.transformer_heads,
                                                         args.transformer_hidden, args.transformer_layers,
                                                         args.dropout_ratio)
        self.dual_ptr_rnn_decoder = DualPtrRNNDecoder(args.dimension, args.dimension,
                                                      dropout=args.dropout_ratio, num_layers=args.rnn_layers)

        self.generative_vocab_size = min(len(field.vocab), args.max_generative_vocab)
        self.out = nn.Linear(args.dimension, self.generative_vocab_size)

        self.dropout = nn.Dropout(0.4)

        self.linear = nn.Linear(args.dimension, 2)



    def set_embeddings(self, embeddings):
        self.encoder_embeddings.set_embeddings(embeddings)
        self.decoder_embeddings.set_embeddings(embeddings)

    def forward(self, batch, iteration):
        context, context_lengths, context_limited = batch.context, batch.context_lengths, batch.context_limited
        question, question_lengths, question_limited = batch.question, batch.question_lengths, batch.question_limited
        answer, answer_lengths, answer_limited = batch.answer, batch.answer_lengths, batch.answer_limited
        oov_to_limited_idx, limited_idx_to_full_idx = batch.oov_to_limited_idx, batch.limited_idx_to_full_idx

        def map_to_full(x):
            return limited_idx_to_full_idx[x]

        self.map_to_full = map_to_full
        context_embedded = self.encoder_embeddings(context)
        question_embedded = self.encoder_embeddings(question)


        context_encoded = self.bilstm_before_coattention(context_embedded, context_lengths)[0]
        question_encoded = self.bilstm_before_coattention(question_embedded, question_lengths)[0]

        context_padding = context.data == self.pad_idx
        question_padding = question.data == self.pad_idx

        coattended_context, coattended_question = self.coattention(context_encoded, question_encoded, context_padding,
                                                                   question_padding)

        context_summary = torch.cat([coattended_context, context_encoded, context_embedded], -1)
        condensed_context, _ = self.context_bilstm_after_coattention(context_summary, context_lengths)
        self_attended_context = self.self_attentive_encoder_context(condensed_context, padding=context_padding)
        final_context, (context_rnn_h, context_rnn_c) = self.bilstm_context(self_attended_context[-1], context_lengths)
        context_rnn_state = [self.reshape_rnn_state(x) for x in (context_rnn_h, context_rnn_c)]

        question_summary = torch.cat([coattended_question, question_encoded, question_embedded], -1)
        condensed_question, _ = self.question_bilstm_after_coattention(question_summary, question_lengths)
        self_attended_question = self.self_attentive_encoder_question(condensed_question, padding=question_padding)
        final_question, (question_rnn_h, question_rnn_c) = self.bilstm_question(self_attended_question[-1],
                                                                                question_lengths)
        question_rnn_state = [self.reshape_rnn_state(x) for x in (question_rnn_h, question_rnn_c)]

        context_indices = context_limited if context_limited is not None else context
        question_indices = question_limited if question_limited is not None else question
        answer_indices = answer_limited if answer_limited is not None else answer

        pad_idx = self.field.decoder_stoi[self.field.pad_token]
        context_padding = context_indices.data == pad_idx
        question_padding = question_indices.data == pad_idx

        self.dual_ptr_rnn_decoder.applyMasks(context_padding, question_padding)

        if self.training:

            context_rnn_h_state = context_rnn_state[0]
            batch_size = context_rnn_h_state.size(0)

            scores = self.linear(context_rnn_h.view(batch_size, 1, -1).squeeze(1))

            probs, targets = mask(answer_indices[:, 1:].contiguous(), probs.contiguous(), pad_idx=pad_idx)

            size = list(outputs.size())

            size[-1] = self.generative_vocab_size
            scores = generator(outputs.view(-1, outputs.size(-1))).view(size)
            p_vocab = F.softmax(scores, dim=scores.dim() - 1)
            scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

            probs = self.probs(self.out, rnn_output, vocab_pointer_switch, context_question_switch,
                               context_attention, question_attention,
                               context_indices, question_indices,
                               oov_to_limited_idx)






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

            probs, targets = mask(answer_indices[:, 1:].contiguous(), probs.contiguous(), pad_idx=pad_idx)

            loss = F.nll_loss(probs.log(), targets)
            return loss, None

        else:
            return None, self.greedy(self_attended_context, final_context, final_question,
                context_indices, question_indices,
                oov_to_limited_idx, rnn_state=context_rnn_state).data



    def reshape_rnn_state(self, h):
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2).contiguous()




