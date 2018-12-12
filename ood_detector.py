#!/usr/bin/env python3
import os
from text import torchtext
from argparse import ArgumentParser
import ujson as json
import torch
import torch.nn.functional as F
import numpy as np
import random
from pprint import pformat

from util import get_splits, set_seed, preprocess_examples, tokenizer
from metrics import compute_metrics
import models
from text.torchtext.data.utils import get_tokenizer

from utils.lang_utils import *

import matplotlib.pyplot as plt
from scipy import interp
from sklearn.utils.fixes import signature

from sklearn import metrics


def get_all_splits(args, new_vocab):
    splits = []
    for task in args.tasks:
        print(f'Loading {task}')
        kwargs = {}
        if not 'train' in args.evaluate:
            kwargs['train'] = None
        if not 'valid' in args.evaluate:
            kwargs['validation'] = None
        if not 'test' in args.evaluate:
            kwargs['test'] = None
        s = get_splits(args, task, new_vocab, **kwargs)[0]
        preprocess_examples(args, [task], [s], new_vocab, train=False)
        splits.append(s)
    return splits


def prepare_data(args, FIELD):
    new_vocab = torchtext.data.SimpleReversibleField(batch_first=True, init_token='<init>', eos_token='<eos>',
                                                     lower=args.lower, include_lengths=True)
    splits = get_all_splits(args, new_vocab)
    new_vocab.build_vocab(*splits)
    print(f'Vocabulary has {len(FIELD.vocab)} tokens from training')
    args.max_generative_vocab = min(len(FIELD.vocab), args.max_generative_vocab)
    FIELD.append_vocab(new_vocab)
    print(f'Vocabulary has expanded to {len(FIELD.vocab)} tokens')

    char_vectors = torchtext.vocab.CharNGram(cache=args.embeddings)
    glove_vectors = torchtext.vocab.GloVe(cache=args.embeddings)
    vectors = [char_vectors, glove_vectors]
    FIELD.vocab.load_vectors(vectors, True)
    FIELD.decoder_to_vocab = {idx: FIELD.vocab.stoi[word] for idx, word in enumerate(FIELD.decoder_itos)}
    FIELD.vocab_to_decoder = {idx: FIELD.decoder_stoi[word] for idx, word in enumerate(FIELD.vocab.itos) if
                              word in FIELD.decoder_stoi}
    splits = get_all_splits(args, FIELD)

    return FIELD, splits


def to_iter(data, bs, device):
    Iterator = torchtext.data.Iterator
    it = Iterator(data, batch_size=bs,
                  device=device, batch_size_fn=None,
                  train=False, repeat=False, sort=None,
                  shuffle=None, reverse=False)

    return it


def run(args, field, val_sets, model):
    device = set_seed(args)
    print(f'Preparing iterators')
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
    print(f'{args.model} has {num_param:,} parameters')
    model.to(device)

    model.eval()
    mode = args.mode

    with torch.no_grad():
        task, it = iters[0]

        # predictions *****
        threshold = args.optimal_th
        scores = []
        predictions = []
        for batch_idx, batch in enumerate(it):

            pred, confidence, penultimate_scores = model(batch, iteration=1, lambd=0)
            confidence = torch.mean(confidence.squeeze(1))

            out = []

            if mode == 'confidence':
                confidence = confidence.data.cpu().numpy()
                out.append(confidence)

            elif mode == 'baseline':
                epsilon = args.epsilon
                # https://arxiv.org/abs/1610.02136
                pred = torch.max(pred.data, 1)[0]
                pred = torch.mean(pred)
                pred = pred.cpu().numpy()
                out.append(pred)

            elif mode == 'temperature':
                T = args.T

                penultimate_scores /= T
                pred = F.softmax(penultimate_scores, dim=-1)
                pred = torch.max(pred.data, 1)[0]
                pred = torch.mean(pred)
                pred = pred.cpu().numpy()
                out.append(pred)


            for ss in out:
                scores.append(float(ss))



        # answers *****

        answers = []
        for batch_idx, batch in enumerate(it):
            if task == 'almond':
                setattr(field, 'use_revtok', False)
                setattr(field, 'tokenize', tokenizer)
                a = field.reverse_almond(batch.answer.data)
                setattr(field, 'use_revtok', True)
                setattr(field, 'tokenize', 'revtok')
            else:
                a = field.reverse(batch.answer.data)
            for aa in a:
                if aa == 'positive':
                    aa = 1
                else:
                    aa = 0
                answers.append(aa)


        ind_scores = [score for ans, score in zip(answers, scores) if ans==1]
        ood_scores = [score for ans, score in zip(answers, scores) if ans==0]


        #######
        scores = np.array(scores)
        ood_scores = np.array(ood_scores)
        ind_scores = np.array(ind_scores)
        answers = np.array(answers)
        fpr_at_95_tpr = tpr95(ind_scores, ood_scores)
        detection_error, best_delta = detection(ind_scores, ood_scores)
        auroc = metrics.roc_auc_score(answers, scores)
        aupr_in = metrics.average_precision_score(answers, scores)
        aupr_out = metrics.average_precision_score(-1 * answers + 1, 1 - scores)
        #######

        print("")
        print("Method: " + args.mode)
        print("TPR95 (lower is better): ", fpr_at_95_tpr)
        print("Detection error (lower is better): ", detection_error)
        print("Best threshold:", best_delta)
        print("AUROC (higher is better): ", auroc)
        print("AUPR_IN (higher is better): ", aupr_in)
        print("AUPR_OUT (higher is better): ", aupr_out)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--evaluate', type=str, required=True)
    parser.add_argument('--tasks',
                        default=['almond', 'squad', 'iwslt.en.de', 'cnn_dailymail', 'multinli.in.out', 'sst', 'srl',
                                 'zre', 'woz.en', 'wikisql', 'schema'], nargs='+')
    parser.add_argument('--devices', default=[0], nargs='+', type=int,
                        help='a list of devices that can be used (multi-gpu currently WIP)')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--data', default='./decaNLP/.data/', type=str, help='where to load data from.')
    parser.add_argument('--embeddings', default='./decaNLP/.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--checkpoint_name')
    parser.add_argument('--bleu', action='store_true', help='whether to use the bleu metric (always on for iwslt)')
    parser.add_argument('--rouge', action='store_true',
                        help='whether to use the bleu metric (always on for cnn, dailymail, and cnn_dailymail)')
    parser.add_argument('--overwrite_predictions', action='store_true',
                        help='whether to overwrite previously written predictions')
    parser.add_argument('--silent', action='store_true', help='whether to print predictions to stdout')

    parser.add_argument('--skip_cache', action='store_true', dest='skip_cache_bool',
                        help='whether to use exisiting cached splits or generate new ones')
    parser.add_argument('--reverse_task', action='store_true', dest='reverse_task_bool',
                        help='whether to translate english to code or the other way around')
    parser.add_argument('--eval_dir', type=str, default=None, help='use this directory to store eval results')

    parser.add_argument('--tune', action='store_true', help='whether to tune or predict')

    parser.add_argument('--test_after_tune', action='store_true',
                        help='test after tuning the model and finding the optimal K and threshold')
    parser.add_argument('--K', default=10, type=int, help='optimal K for topk')
    parser.add_argument('--optimal_th', default=0.5, type=float, help='optimal threshold to gain highest TP/FP')

    parser.add_argument('--mode', default='confidence', type=str, help='cofidence loss regularization strength')
    parser.add_argument('--T', type=float, default=1000., help='Scaling temperature')
    parser.add_argument('--epsilon', type=float, default=0.001, help='Noise magnitude')


    args = parser.parse_args()

    with open(os.path.join(args.path, 'config.json')) as config_file:
        config = json.load(config_file)
        retrieve = ['model',
                    'transformer_layers', 'rnn_layers', 'transformer_hidden',
                    'dimension', 'load', 'max_val_context_length', 'val_batch_size',
                    'transformer_heads', 'max_output_length', 'max_generative_vocab',
                    'lower', 'cove', 'intermediate_cove', 'elmo']
        for r in retrieve:
            if r in config:
                setattr(args, r, config[r])
            elif 'cove' in r or 'elmo' in r:
                setattr(args, r, False)
            else:
                setattr(args, r, None)
        args.dropout_ratio = 0.0
        args.val_batch_size = [1]

    args.task_to_metric = {
        'cnn_dailymail': 'avg_rouge',
        'iwslt.en.de': 'bleu',
        'multinli.in.out': 'em',
        'squad': 'nf1',
        'srl': 'nf1',
        'almond': 'bleu' if args.reverse_task_bool else 'em',
        'sst': 'em',
        'wikisql': 'lfem',
        'woz.en': 'joint_goal_em',
        'zre': 'corpus_f1',
        'schema': 'em'
    }

    if os.path.exists(os.path.join(args.path, 'process_0.log')):
        args.best_checkpoint = get_best(args)
    else:
        args.best_checkpoint = os.path.join(args.path, args.checkpoint_name)

    return args


def get_best(args):
    with open(os.path.join(args.path, 'config.json')) as f:
        save_every = json.load(f)['save_every']

    with open(os.path.join(args.path, 'process_0.log')) as f:
        lines = f.readlines()

    best_score = 0
    best_it = 10
    deca_scores = {}
    for l in lines:
        if 'val' in l:
            try:
                task = l.split('val_')[1].split(':')[0]
            except Exception as e:
                print(e)
                continue
            it = int(l.split('iteration_')[1].split(':')[0])
            metric = args.task_to_metric[task]
            score = float(l.split(metric + '_')[1].split(':')[0])
            if it in deca_scores:
                deca_scores[it]['deca'] += score
                deca_scores[it][metric] = score
            else:
                deca_scores[it] = {'deca': score, metric: score}
            if deca_scores[it]['deca'] > best_score:
                best_score = deca_scores[it]['deca']
                best_it = it
    print(best_it)
    print(best_score)
    return os.path.join(args.path, f'iteration_{int(best_it)}.pth')


if __name__ == '__main__':
    args = get_args()
    print(f'Arguments:\n{pformat(vars(args))}')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f'Loading from {args.best_checkpoint}')

    if torch.cuda.is_available():
        save_dict = torch.load(args.best_checkpoint)
    else:
        save_dict = torch.load(args.best_checkpoint, map_location='cpu')

    field = save_dict['field']
    print(f'Initializing Model')
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
    model.set_embeddings(field.vocab.vectors)

    run(args, field, splits, model)
