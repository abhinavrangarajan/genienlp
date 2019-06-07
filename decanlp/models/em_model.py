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
'''
Created on June 4, 2019

author: mehrad
'''

import itertools

from ..util import set_seed
from decanlp.utils.model_utils import init_model
from .common import *
import numpy

import logging

_logger = logging.getLogger(__name__)

class EM_model(nn.Module):

    def __init__(self, field, args):
        super().__init__()
        self.field = field
        self.args = args
        self.device = set_seed(args)

        if not self.args.use_google_translate:
            names = ['machine_translation', 'semantic_parser', 'thingtalk_machine_translation']
        else:
            names = ['semantic_parser', 'thingtalk_machine_translation']

        for name in names:
            model_dict = torch.load(os.path.join(args.saved_models, f'{name}/best.pth'), map_location=self.device)
            model_field = model_dict['field']
            model = init_model(args, model_field, _logger, args.world_size, self.device, model_name='MultitaskQuestionAnsweringNetwork')
            model.load_state_dict(model_dict['model_state_dict'])
            # model_opt_dict = torch.load(os.path.join(args.saved_models, f'{model}/best_optim.pth'), map_location=self.device)
            # model_opt = init_opt(args, model)
            # model_opt.load_state_dict(model_opt_dict)

            setattr(self, f'{name}_field', model_field)
            setattr(self, f'{name}_model', model)
            # setattr(self, f'{name}_opt', model_opt)

    def forward(self, batch, iteration):
        context, context_lengths, context_limited, context_tokens     = batch.context,  batch.context_lengths,  batch.context_limited, batch.context_tokens
        question, question_lengths, question_limited, question_tokens = batch.question, batch.question_lengths, batch.question_limited, batch.question_tokens
        answer, answer_lengths, answer_limited, answer_tokens         = batch.answer,   batch.answer_lengths,   batch.answer_limited, batch.answer_tokens
        oov_to_limited_idx, limited_idx_to_full_idx  = batch.oov_to_limited_idx, batch.limited_idx_to_full_idx

        setattr(batch, 'context_limited', None)
        setattr(batch, 'question_limited', None)
        setattr(batch, 'answer_limited', None)

        original_context = context
        original_question = question

        batch_size = len(question_tokens)

        def make_one_hot(tensor, C):
            one_hot = torch.zeros([*tensor.size(), C])
            target = one_hot.scatter_(-1, tensor.unsqueeze(-1), 1)

            return target

        def sample_sentence(probs):

            B, T, C = probs.size()
            probs_flattened = probs.reshape([B*T, C])
            sample = torch.tensor(([np.random.choice(C, p=row.detach().numpy()) for row in probs_flattened]))
            mask = make_one_hot(sample, C)

            probs = torch.sum(probs_flattened*mask, -1)

            return sample.reshape([B, T]), probs.reshape([B, T])


        # running through Farsi-English MT
        #### TODO fix this later
        if not self.args.use_google_translate:
            # question = 'Translate from Farsi to English'.split()
            question = 'Translate from English to ThingTalk'.split()
            question_tokens_single = [self.machine_translation_field.init_token] + question + [self.machine_translation_field.eos_token]
            question_tokens = list(itertools.repeat(question_tokens_single, batch_size))
            context_numericalized = [[self.machine_translation_field.decoder_stoi.get(sentence[time], self.machine_translation_field.decoder_stoi['<unk>']) for sentence in context_tokens] for time in range(len(context_tokens[0]))]
            question_numericalized = [[self.machine_translation_field.decoder_stoi[sentence[time]] for sentence in question_tokens] for time in range(len(question_tokens[0]))]

            context = torch.tensor(context_numericalized, dtype=torch.long, device=self.device).t_()
            question = torch.tensor(question_numericalized, dtype=torch.long, device=self.device).t_()
            setattr(batch, 'context', context)
            setattr(batch, 'question', question)


            self.machine_translation_model.train()
            _, probs_output_from_MT = self.machine_translation_model(batch, iteration)
            sample_output_from_MT, sample_prob_output_from_MT = sample_sentence(probs_output_from_MT)
            task = self.machine_translation_model.args.train_tasks[0] if hasattr(self.machine_translation_model.args, 'train_tasks') else self.machine_translation_model.args.tasks[0]

            sampled_output_from_MT_tokens = self.machine_translation_field.reverse(sample_output_from_MT, detokenize=task.detokenize, field_name='answer')


        # numericalize input to semantic parser
        question = 'Translate from English to ThingTalk'.split()
        question_tokens_single = [self.semantic_parser_field.init_token] + question + [self.semantic_parser_field.eos_token]
        question_tokens = list(itertools.repeat(question_tokens_single, batch_size))

        if not self.args.use_google_translate:
            context_tokens = [[self.semantic_parser_field.init_token] + tokens.split(' ') + [self.semantic_parser_field.eos_token] for tokens in sampled_output_from_MT_tokens]
            context_lengths = [len(sublist) for sublist in context_tokens]
            max_length = max(context_lengths)
            context_tokens = [tokens + [self.semantic_parser_field.pad_token] * (max_length - len(tokens)) for tokens in context_tokens]


        context_numericalized = [[self.semantic_parser_field.decoder_stoi.get(sentence[time], self.semantic_parser_field.decoder_stoi['<unk>']) for sentence in context_tokens] for time in range(len(context_tokens[0]))]
        question_numericalized = [[self.semantic_parser_field.decoder_stoi[sentence[time]] for sentence in question_tokens] for time in range(len(question_tokens[0]))]
        context = torch.tensor(context_numericalized, dtype=torch.long, device=self.device, requires_grad=False).t_()
        question = torch.tensor(question_numericalized, dtype=torch.long, device=self.device, requires_grad=False).t_()
        setattr(batch, 'context_tokens', context_tokens)
        setattr(batch, 'context', context)
        setattr(batch, 'context_lengths', context_lengths)
        setattr(batch, 'question', question)

        self.semantic_parser_model.train()
        # _, greedy_output_from_SP = self.semantic_parser_model(batch, iteration)
        _, probs_output_from_SP = self.semantic_parser_model(batch, iteration)
        sample_output_from_SP, sample_prob_output_from_SP = sample_sentence(probs_output_from_SP)
        task = self.semantic_parser_model.args.train_tasks[0] if hasattr(self.semantic_parser_model.args, 'train_tasks') else self.semantic_parser_model.args.tasks[0]
        sample_output_from_SP_tokens = self.semantic_parser_field.reverse(sample_output_from_SP, detokenize=task.detokenize, field_name='answer')

        print(f'**** greedy_output_from_thingtalk_MT_tokens: {sample_output_from_SP_tokens[0]} ***')

        # numericalize input to thingtalk machine translation
        #### TODO fix this later
        # question = 'Translate from English to Farsi'.split()
        question = 'Translate from English to ThingTalk'.split()
        question_tokens_single = [self.semantic_parser_field.init_token] + question + [self.semantic_parser_field.eos_token]
        question_tokens = list(itertools.repeat(question_tokens_single, batch_size))

        context_tokens = [[self.semantic_parser_field.init_token] + tokens.split(' ') + [self.semantic_parser_field.eos_token] for tokens in sample_output_from_SP_tokens]
        context_lengths = [len(sublist) for sublist in context_tokens]
        max_length = max(context_lengths)
        context_tokens = [tokens + [self.thingtalk_machine_translation_field.pad_token] * (max_length - len(tokens)) for tokens in context_tokens]

        context_numericalized = [[self.thingtalk_machine_translation_field.decoder_stoi.get(sentence[time], self.thingtalk_machine_translation_field.decoder_stoi['<unk>']) for sentence in context_tokens] for time in range(len(context_tokens[0]))]
        question_numericalized = [[self.thingtalk_machine_translation_field.decoder_stoi[sentence[time]] for sentence in question_tokens] for time in range(len(question_tokens[0]))]

        ##  TODO fix this
        answer_numericalized = [[self.field.decoder_stoi.get(sentence[time], self.field.decoder_stoi['<unk>']) for sentence in answer_tokens] for time in range(len(answer_tokens[0]))]

        context = torch.tensor(context_numericalized, dtype=torch.long, device=self.device, requires_grad=False).t_()
        question = torch.tensor(question_numericalized, dtype=torch.long, device=self.device, requires_grad=False).t_()
        answer = torch.tensor(answer_numericalized, dtype=torch.long, device=self.device, requires_grad=False).t_()
        setattr(batch, 'context_tokens', context_tokens)
        setattr(batch, 'context', context)
        setattr(batch, 'context_lengths', context_lengths)
        setattr(batch, 'question', question)
        setattr(batch, 'answer', answer)

        if self.training:

            self.thingtalk_machine_translation_model.train()
            loss, _ = self.thingtalk_machine_translation_model(batch, iteration)
            # loss, _ = self.thingtalk_machine_translation_model(batch, iteration)
            # _, probs_output_from_thingtalk_MT = self.machine_translation_model(batch, iteration)
            # sample_output_from_thingtalk_MT, sample_prob_output_from_thingtalk_MT = sample_sentence(probs_output_from_thingtalk_MT)


            r = self.args.alpha * torch.mean(sample_prob_output_from_SP) + (1 - self.args.alpha) * loss
            first_loss = r * torch.mean(sample_prob_output_from_SP)
            second_loss = (1 - self.args.alpha) * loss

            return first_loss, second_loss

        else:

            self.thingtalk_machine_translation_model.eval()
            _, greedy_output_from_thingtalk_MT = self.thingtalk_machine_translation_model(batch, iteration)
            task = self.thingtalk_machine_translation_model.args.train_tasks[0] if hasattr(self.thingtalk_machine_translation_model.args, 'train_tasks') else self.thingtalk_machine_translation_model.args.tasks[0]
            greedy_output_from_thingtalk_MT_tokens = self.thingtalk_machine_translation_field.reverse(greedy_output_from_thingtalk_MT, detokenize=task.detokenize, field_name='answer')
            print(f'**** greedy_output_from_thingtalk_MT_tokens: {greedy_output_from_thingtalk_MT_tokens[0]} ***')

            context_tokens = [[self.thingtalk_machine_translation_field.init_token] + tokens.split(' ') + [self.thingtalk_machine_translation_field.eos_token] for tokens in greedy_output_from_thingtalk_MT_tokens]
            context_lengths = [len(sublist) for sublist in context_tokens]
            max_length = max(context_lengths)
            context_tokens = [tokens + [self.thingtalk_machine_translation_field.pad_token] * (max_length - len(tokens)) for tokens in context_tokens]

            prediction_tokens = context_tokens
            prediction_numericalized = [[self.field.decoder_stoi.get(sentence[time], self.field.decoder_stoi['<unk>']) for sentence in prediction_tokens] for time in range(len(prediction_tokens[0]))]
            prediction = torch.tensor(prediction_numericalized, dtype=torch.long, device=self.device).t_()

            setattr(batch, 'context', original_context)
            setattr(batch, 'question', original_question)

            return None, prediction

