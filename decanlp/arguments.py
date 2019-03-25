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
from copy import deepcopy
import types
import sys
from argparse import ArgumentParser
import subprocess
import json
import datetime
from dateutil import tz
import logging

from .tasks.registry import get_tasks

logger = logging.getLogger(__name__)

def get_commit():
    directory = os.path.dirname(__file__)
    return subprocess.Popen("cd {} && git log | head -n 1".format(directory), shell=True, stdout=subprocess.PIPE).stdout.read().split()[1].decode()


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def parse(argv):
    """
    Returns the arguments from the command line.
    """
    parser = ArgumentParser(prog=argv[0])
    parser.add_argument('--root', default='./decaNLP', type=str, help='root directory for data, results, embeddings, code, etc.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--save', default='results', type=str, help='where to save results.')
    parser.add_argument('--embeddings', default='.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--cached', default='', type=str, help='where to save cached files')

    parser.add_argument('--train_tasks', nargs='+', type=str, dest='train_task_names', help='tasks to use for training', required=True)
    parser.add_argument('--train_iterations', nargs='+', type=int, help='number of iterations to focus on each task')
    parser.add_argument('--train_batch_tokens', nargs='+', default=[9000], type=int, help='Number of tokens to use for dynamic batching, corresponging to tasks in train tasks')
    parser.add_argument('--jump_start', default=0, type=int, help='number of iterations to give jump started tasks')
    parser.add_argument('--n_jump_start', default=0, type=int, help='how many tasks to jump start (presented in order)')    
    parser.add_argument('--num_print', default=15, type=int, help='how many validation examples with greedy output to print to std out')

    parser.add_argument('--no_tensorboard', action='store_false', dest='tensorboard', help='Turn of tensorboard logging')
    parser.add_argument('--max_to_keep', default=5, type=int, help='number of checkpoints to keep')
    parser.add_argument('--log_every', default=int(1e2), type=int, help='how often to log results in # of iterations')
    parser.add_argument('--save_every', default=int(1e3), type=int, help='how often to save a checkpoint in # of iterations')

    parser.add_argument('--val_tasks', nargs='+', type=str, dest='val_task_names', help='tasks to collect evaluation metrics for')
    parser.add_argument('--val_every', default=int(1e3), type=int, help='how often to run validation in # of iterations')
    parser.add_argument('--val_no_filter', action='store_false', dest='val_filter', help='whether to allow filtering on the validation sets')
    parser.add_argument('--val_batch_size', nargs='+', default=[256], type=int, help='Batch size for validation corresponding to tasks in val tasks')

    parser.add_argument('--vocab_tasks', nargs='+', type=str, help='tasks to use in the construction of the vocabulary')
    parser.add_argument('--max_output_length', default=100, type=int, help='maximum output length for generation')
    parser.add_argument('--max_effective_vocab', default=int(1e6), type=int, help='max effective vocabulary size for pretrained embeddings')
    parser.add_argument('--max_generative_vocab', default=50000, type=int, help='max vocabulary for the generative softmax')
    parser.add_argument('--max_train_context_length', default=500, type=int, help='maximum length of the contexts during training')
    parser.add_argument('--max_val_context_length', default=500, type=int, help='maximum length of the contexts during validation')
    parser.add_argument('--max_answer_length', default=50, type=int, help='maximum length of answers during training and validation')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the datasets')
    parser.add_argument('--preserve_case', action='store_false', dest='lower', help='whether to preserve casing for all text')

    parser.add_argument('--model', type=str, default='MultitaskQuestionAnsweringNetwork', help='which model to import')
    parser.add_argument('--dimension', default=200, type=int, help='output dimensions for all layers')
    parser.add_argument('--rnn_layers', default=1, type=int, help='number of layers for RNN modules')
    parser.add_argument('--transformer_layers', default=2, type=int, help='number of layers for transformer modules')
    parser.add_argument('--transformer_hidden', default=150, type=int, help='hidden size of the transformer modules')
    parser.add_argument('--transformer_heads', default=3, type=int, help='number of heads for transformer modules')
    parser.add_argument('--dropout_ratio', default=0.2, type=float, help='dropout for the model')
    parser.add_argument('--cove', action='store_true', help='whether to use contextualized word vectors (McCann et al. 2017)')
    parser.add_argument('--intermediate_cove', action='store_true', help='whether to use the intermediate layers of contextualized word vectors (McCann et al. 2017)')
    parser.add_argument('--elmo', default=[-1], nargs='+', type=int,  help='which layer(s) (0, 1, or 2) of ELMo (Peters et al. 2018) to use; -1 for none ')
    parser.add_argument('--no_glove_and_char', action='store_false', dest='glove_and_char', help='turn off GloVe and CharNGram embeddings')
    parser.add_argument('--trainable_decoder_embedding', default=0, type=int, help='size of trainable portion of decoder embedding (0 or omit to disable)')
    parser.add_argument('--no_glove_decoder', action='store_false', dest='glove_decoder', help='turn off GloVe embeddings from decoder')

    parser.add_argument('--warmup', default=800, type=int, help='warmup for learning rate')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping')
    parser.add_argument('--beta0', default=0.9, type=float, help='alternative momentum for Adam (only when not using transformer_lr)')
    parser.add_argument('--optimizer', default='adam', type=str, help='Adam or SGD')
    parser.add_argument('--no_transformer_lr', action='store_false', dest='transformer_lr', help='turns off the transformer learning rate strategy') 
    parser.add_argument('--sgd_lr', default=1.0, type=float, help='learning rate for SGD (if not using Adam)')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight L2 regularization')

    parser.add_argument('--load', default=None, type=str, help='path to checkpoint to load model from inside args.save')
    parser.add_argument('--resume', action='store_true', help='whether to resume training with past optimizers')

    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used for training (multi-gpu currently WIP)')
    parser.add_argument('--backend', default='gloo', type=str, help='backend for distributed training')

    parser.add_argument('--no_commit', action='store_false', dest='commit', help='do not track the git commit associated with this training run') 
    parser.add_argument('--exist_ok', action='store_true', help='Ok if the save directory already exists, i.e. overwrite is ok') 
    parser.add_argument('--token_testing', action='store_true', help='if true, sorts all iterators') 
    parser.add_argument('--reverse', action='store_true', help='if token_testing and true, sorts all iterators in reverse') 

    parser.add_argument('--skip_cache', action='store_true', dest='skip_cache_bool', help='whether to use exisiting cached splits or generate new ones')
    parser.add_argument('--lr_rate', default=0.001, type=float, help='initial_learning_rate')
    parser.add_argument('--use_bleu_loss', action='store_true', help='whether to use differentiable BLEU loss or not')
    parser.add_argument('--use_maxmargin_loss', action='store_true', help='whether to use max-margin loss or not')
    parser.add_argument('--loss_switch', default=0.666, type=float, help='switch to BLEU loss after certain iterations controlled by this ratio')
    parser.add_argument('--small_glove', action='store_true', help='Use glove.6B.50d instead of glove.840B.300d')
    parser.add_argument('--almond_type_embeddings', action='store_true', help='Add type-based word embeddings for Almond task')
    parser.add_argument('--use_curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--aux_dataset', default='', type=str, help='path to auxiliary dataset (ignored if curriculum is not used)')
    parser.add_argument('--curriculum_max_frac', default=1.0, type=float, help='max fraction of harder dataset to keep for curriculum')
    parser.add_argument('--curriculum_rate', default=0.1, type=float, help='growth rate for curriculum')
    parser.add_argument('--curriculum_strategy', default='linear', type=str, choices=['linear', 'exp'], help='growth strategy for curriculum')
    parser.add_argument('--thingpedia', type=str, help='where to load thingpedia.json from (for almond task only)')
    parser.add_argument('--almond_grammar', type=str,
                        choices=['typeless.bottomup', 'typeless.topdown', 'plain.bottomup', 'plain.topdown', 'pos.typeless.bottomup', 'pos.typeless.topdown',
                                 'pos.bottomup', 'pos.topdown', 'full.bottomup', 'full.topdown'],
                        help="which grammar to use for Almond task (leave unspecified for no grammar)")

    args = parser.parse_args(argv[1:])
    if args.model is None:
        args.model = 'mcqa'

    if args.val_task_names is None:
        args.val_task_names = []
        for t in args.train_task_names:
            if t not in args.val_task_names:
                args.val_task_names.append(t)
    if 'imdb' in args.val_task_names:
        args.val_task_names.remove('imdb')

    args.world_size = len(args.devices) if args.devices[0] > -1 else -1
    if args.world_size > 1:
        logger.error('multi-gpu training is currently a work in progress')
        return
    args.timestamp = '-'.join(datetime.datetime.now(tz=tz.tzoffset(None, -8*60*60)).strftime("%y/%m/%d/%H/%M/%S.%f").split())

    if len(args.train_task_names) > 1:
        if args.train_iterations is  None:
            args.train_iterations = [1]
        if len(args.train_iterations) < len(args.train_task_names):
            args.train_iterations = len(args.train_task_names) * args.train_iterations
        if len(args.train_batch_tokens) < len(args.train_task_names):
            args.train_batch_tokens = len(args.train_task_names) * args.train_batch_tokens
    if len(args.val_batch_size) < len(args.val_task_names):
        args.val_batch_size = len(args.val_task_names) * args.val_batch_size
        
    # postprocess arguments
    if args.commit:
        args.commit = get_commit()
    else:
        args.commit = ''

    args.log_dir = args.save
    args.dist_sync_file = os.path.join(args.log_dir, 'distributed_sync_file')
    
    for x in ['data', 'save', 'embeddings', 'log_dir', 'dist_sync_file']:
        setattr(args, x, os.path.join(args.root, getattr(args, x)))
    save_args(args)

    # create the task objects after we saved the configuration to the JSON file, because
    # tasks are not JSON serializable
    args.train_tasks = get_tasks(args.train_task_names, args)
    args.val_tasks = get_tasks(args.val_task_names, args)

    return args
