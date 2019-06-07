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
from .text import torchtext
from argparse import ArgumentParser
import ujson as json
import torch
import numpy as np
import random
import sys
import logging
from pprint import pformat

from .util import set_seed, preprocess_examples, load_config_json
from .metrics import compute_metrics
from .utils.embeddings import load_embeddings
from .tasks.registry import get_tasks
from . import models

logger = logging.getLogger(__name__)

def get_all_splits(args, new_vocab):
    splits = []
    for task in args.tasks:
        logger.info(f'Loading {task}')
        kwargs = {}
        if not 'train' in args.evaluate:
            kwargs['train'] = None
        if not 'valid' in args.evaluate:
            kwargs['validation'] = None
        if not 'test' in args.evaluate:
            kwargs['test'] = None

        kwargs['skip_cache_bool'] = args.skip_cache_bool
        kwargs['cached_path'] = args.cached
        kwargs['subsample'] = args.subsample
        s = task.get_splits(new_vocab, root=args.data, **kwargs)[0]
        preprocess_examples(args, [task], [s], new_vocab, train=False)
        splits.append(s)
    return splits


def prepare_data(args, FIELD):
    new_vocab = torchtext.data.ReversibleField(batch_first=True, init_token='<init>', eos_token='<eos>', lower=args.lower, include_lengths=True)
    splits = get_all_splits(args, new_vocab)
    new_vocab.build_vocab(*splits)
    logger.info(f'Vocabulary has {len(FIELD.vocab)} tokens from training')
    args.max_generative_vocab = min(len(FIELD.vocab), args.max_generative_vocab)
    FIELD.append_vocab(new_vocab)
    logger.info(f'Vocabulary has expanded to {len(FIELD.vocab)} tokens')
    vectors = load_embeddings(args)
    FIELD.vocab.load_vectors(vectors, True)
    FIELD.decoder_to_vocab = {idx: FIELD.vocab.stoi[word] for idx, word in enumerate(FIELD.decoder_itos)}
    FIELD.vocab_to_decoder = {idx: FIELD.decoder_stoi[word] for idx, word in enumerate(FIELD.vocab.itos) if word in FIELD.decoder_stoi}
    splits = get_all_splits(args, FIELD)

    return FIELD, splits


def to_iter(data, bs, device):
    Iterator = torchtext.data.Iterator
    it = Iterator(data, batch_size=bs, 
       device=device, batch_size_fn=None, 
       train=False, repeat=False, sort=False,
       shuffle=False, reverse=False)

    return it


def run(args, field, val_sets, model):
    device = set_seed(args)
    logger.info(f'Preparing iterators')
    if len(args.val_batch_size) == 1 and len(val_sets) > 1:
        args.val_batch_size *= len(val_sets)
    iters = [(name, to_iter(x, bs, device)) for name, x, bs in zip(args.tasks, val_sets, args.val_batch_size)]
 
    def mult(ps):
        r = 0
        for p in ps:
            this_r = 1
            for s in p.size():
                this_r *= s
            r += this_r
        return r
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_param = mult(params)
    logger.info(f'{args.model} has {num_param:,} parameters')
    model.to(device)

    decaScore = []
    model.eval()
    setattr(model, 'prediction', True)
    with torch.no_grad():
        for task, it in iters:
            logger.info(task.name)
            if args.eval_dir:
                prediction_file_name = os.path.join(args.eval_dir, os.path.join(args.evaluate, task.name + '.txt'))
                answer_file_name = os.path.join(args.eval_dir, os.path.join(args.evaluate, task.name + '.gold.txt'))
                results_file_name = answer_file_name.replace('gold', 'results')
                context_file_name = os.path.join(args.eval_dir, os.path.join(args.evaluate, task.name + '.context.txt'))
            else:
                prediction_file_name = os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task.name + '.txt')
                answer_file_name = os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task.name + '.gold.txt')
                results_file_name = answer_file_name.replace('gold', 'results')
                context_file_name = os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task.name + '.context.txt')
            if 'sql' in task.name or 'squad' in task.name:
                ids_file_name = answer_file_name.replace('gold', 'ids')
            if os.path.exists(prediction_file_name):
                logger.warning(f'** {prediction_file_name} already exists -- this is where predictions are stored **')
                if args.overwrite:
                    logger.warning(f'**** overwriting {prediction_file_name} ****')
            if os.path.exists(answer_file_name):
                logger.warning(f'** {answer_file_name} already exists -- this is where ground truth answers are stored **')
                if args.overwrite:
                    logger.warning(f'**** overwriting {answer_file_name} ****')
            if os.path.exists(context_file_name):
                logger.warning(f'** {context_file_name} already exists -- this is where context sentences are stored **')
                if args.overwrite:
                    logger.warning(f'**** overwriting {context_file_name} ****')
            if os.path.exists(results_file_name):
                logger.warning(f'** {results_file_name} already exists -- this is where metrics are stored **')
                if args.overwrite:
                    logger.warning(f'**** overwriting {results_file_name} ****')
                else:
                    lines = open(results_file_name).readlines()
                    if not args.silent:
                        for l in lines:
                            logger.warning(l)
                    metrics = json.loads(lines[0])
                    decaScore.append(metrics[task.metrics[0]])
                    continue

            for x in [prediction_file_name, answer_file_name, results_file_name, context_file_name]:
                os.makedirs(os.path.dirname(x), exist_ok=True)

            if not os.path.exists(prediction_file_name) or args.overwrite:
                with open(prediction_file_name, 'w') as prediction_file:
                    predictions = []
                    ids = []
                    for batch_idx, batch in enumerate(it):
                        _, p = model(batch, iteration=1)

                        p = field.reverse(p, detokenize=task.detokenize, field_name='answer')

                        for i, pp in enumerate(p):
                            if 'sql' in task.name:
                                ids.append(int(batch.wikisql_id[i]))
                            if 'squad' in task.name:
                                ids.append(it.dataset.q_ids[int(batch.squad_id[i])])
                            prediction_file.write(pp + '\n')
                            predictions.append(pp)
                if 'sql' in task.name:
                    with open(ids_file_name, 'w') as id_file:
                        for i in ids:
                            id_file.write(json.dumps(i) + '\n')
                if 'squad' in task.name:
                    with open(ids_file_name, 'w') as id_file:
                        for i in ids:
                            id_file.write(i + '\n')
            else:
                with open(prediction_file_name) as prediction_file:
                    predictions = [x.strip() for x in prediction_file.readlines()]
                if 'sql' in task.name or 'squad' in task.name:
                    with open(ids_file_name) as id_file:
                        ids = [int(x.strip()) for x in id_file.readlines()]

            def from_all_answers(an):
                return [it.dataset.all_answers[sid] for sid in an.tolist()]

            if not os.path.exists(answer_file_name) or args.overwrite:
                with open(answer_file_name, 'w') as answer_file:
                    answers = []
                    for batch_idx, batch in enumerate(it):
                        if hasattr(batch, 'wikisql_id'):
                            a = from_all_answers(batch.wikisql_id.data.cpu())
                        elif hasattr(batch, 'squad_id'):
                            a = from_all_answers(batch.squad_id.data.cpu())
                        elif hasattr(batch, 'woz_id'):
                            a = from_all_answers(batch.woz_id.data.cpu())
                        else:
                            a = field.reverse(batch.answer.data, detokenize=task.detokenize, field_name='answer')
                        for aa in a:
                            answers.append(aa)
                            answer_file.write(aa + '\n')
            else:
                with open(answer_file_name) as answer_file:
                    answers = [json.loads(x.strip()) for x in answer_file.readlines()]

            if not os.path.exists(context_file_name) or args.overwrite:
                with open(context_file_name, 'w') as context_file:
                    contexts = []
                    for batch_idx, batch in enumerate(it):
                        c = field.reverse(batch.context.data, detokenize=task.detokenize, field_name='context')
                        for cc in c:
                            contexts.append(cc)
                            context_file.write(cc + '\n')

            if len(answers) > 0:
                if not os.path.exists(results_file_name) or args.overwrite:
                    metrics, answers = compute_metrics(predictions, answers, task.metrics, args=args)
                    with open(results_file_name, 'w') as results_file:
                        results_file.write(json.dumps(metrics) + '\n')
                else:
                    with open(results_file_name) as results_file:
                        metrics = json.loads(results_file.readlines()[0])

                if not args.silent:
                    for i, (p, a) in enumerate(zip(predictions, answers)):
                        logger.info(f'Prediction {i+1}: {p}\nAnswer {i+1}: {a}\n')
                    logger.info(metrics)
                decaScore.append(metrics[task.metrics[0]])

    logger.info(f'Evaluated Tasks:\n')
    for i, (task, _) in enumerate(iters):
        logger.info(f'{task.name}: {decaScore[i]}')
    logger.info(f'-------------------')
    logger.info(f'DecaScore:  {sum(decaScore)}\n')
    logger.info(f'\nSummary: | {sum(decaScore)} | {" | ".join([str(x) for x in decaScore])} |\n')


def get_args(argv):
    parser = ArgumentParser(prog=argv[0])
    parser.add_argument('--path', required=True)
    parser.add_argument('--evaluate', type=str, required=True)
    parser.add_argument('--tasks', default=['almond', 'squad', 'iwslt.en.de', 'cnn_dailymail', 'multinli.in.out', 'sst','srl', 'zre', 'woz.en', 'wikisql', 'schema'], dest='task_names', nargs='+')
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used (multi-gpu currently WIP)')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--data', default='./decaNLP/.data/', type=str, help='where to load data from.')
    parser.add_argument('--embeddings', default='./decaNLP/.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--checkpoint_name', default='best.pth', help='Checkpoint file to use (relative to --path, defaults to best.pth)')
    parser.add_argument('--bleu', action='store_true', help='whether to use the bleu metric (always on for iwslt)')
    parser.add_argument('--rouge', action='store_true', help='whether to use the bleu metric (always on for cnn, dailymail, and cnn_dailymail)')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite previously written predictions')
    parser.add_argument('--silent', action='store_true', help='whether to print predictions to stdout')

    parser.add_argument('--skip_cache', action='store_true', dest='skip_cache_bool', help='whether use exisiting cached splits or generate new ones')
    parser.add_argument('--eval_dir', type=str, default=None, help='use this directory to store eval results')
    parser.add_argument('--cached', default='', type=str, help='where to save cached files')
    parser.add_argument('--thingpedia', type=str, help='where to load thingpedia.json from (for almond task only)')

    parser.add_argument('--beam_search', action='store_true', help='use beam search instead of greedy search during prediction')
    parser.add_argument('--beam_size', default=4, type=int, help='beam_size for beam search')
    parser.add_argument('--saved_models', default='./saved_models', type=str, help='directory where cached models should be loaded from')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the eval/test datasets (experimental)')

    args = parser.parse_args(argv[1:])

    load_config_json(args)
    args.tasks = get_tasks(args.task_names, args)
    return args


def main(argv=sys.argv):
    args = get_args(argv)
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info(f'Loading from {args.best_checkpoint}')

    if torch.cuda.is_available():
        save_dict = torch.load(args.best_checkpoint)
    else:
        save_dict = torch.load(args.best_checkpoint, map_location='cpu')

    field = save_dict['field']
    logger.info(f'Initializing Model')
    Model = getattr(models, args.model)
    model = Model(field, args)
    model_dict = save_dict['model_state_dict']
    backwards_compatible_cove_dict = {}
    for k, v in model_dict.items():
        if 'cove.rnn.' in k:
            k = k.replace('cove.rnn.', 'cove.rnn1.')
        backwards_compatible_cove_dict[k] = v
    model_dict = backwards_compatible_cove_dict
    model.load_state_dict(model_dict)
    field, splits = prepare_data(args, field)
    if args.model != 'MultiLingualTranslationModel':
        model.set_embeddings(field.vocab.vectors)

    run(args, field, splits, model)

if __name__ == '__main__':
    main()
