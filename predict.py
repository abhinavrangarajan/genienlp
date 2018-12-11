#!/usr/bin/env python3
import os
from text import torchtext
from argparse import ArgumentParser
import ujson as json
import torch
import numpy as np
import random
from pprint import pformat

from util import get_splits, set_seed, preprocess_examples, tokenizer
from metrics import compute_metrics
import models
from text.torchtext.data.utils import get_tokenizer

import matplotlib.pyplot as plt
from scipy import interp
from sklearn.utils.fixes import signature


from sklearn.metrics import *


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
    new_vocab = torchtext.data.SimpleReversibleField(batch_first=True, init_token='<init>', eos_token='<eos>', lower=args.lower, include_lengths=True)
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
    FIELD.vocab_to_decoder = {idx: FIELD.decoder_stoi[word] for idx, word in enumerate(FIELD.vocab.itos) if word in FIELD.decoder_stoi}
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

    if args.tune:
        best_K= 1
        best_roc = 0
        best_answers = []
        best_scores = []

        # for K in range(1, args.max_output_length):
        for K in range(1, 4):
            with torch.no_grad():
                task, it = iters[0]

                # predictions *****
                scores = []
                for batch_idx, batch in enumerate(it):
                    _, score = model(batch, iteration=1)
                    _, seq_len = score.size()
                    # score[(score != 1.0).nonzero()
                    score_cleaned = torch.where(score < 0.99999, score, torch.zeros_like(score))

                    # score_sorted, score_sorted_indices = torch.sort(score, dim=1, descending=True)
                    # score = torch.mean(score_sorted[:, int(1/K * seq_len): int((1-1/K) * seq_len)], dim=1)

                    score = torch.mean(torch.topk(score_cleaned, K, dim=1)[0], dim=1)
                    # score = torch.max(score, dim=1)[0]
                    for i, ss in enumerate(score):
                        scores.append(ss)


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



                # ans_sorted, score_sorted = list(zip(*sorted(zip(answers, scores), key= lambda x: x[1], reverse=True)))
                # print(f'-------------------')
                # print(f'answers:  {ans_sorted}\n')
                # print(f'scores: {score_sorted}\n')
                # print(f'-------------------')

                # precision_recall_curve


            area_under_roc = roc_auc_score(answers, scores)

            print(f'-------------------')
            # print(f'precision:  {precision}\n')
            # print(f'recall:  {recall}\n')
            # print(f'thresholds:  {thresholds}\n')
            print(f'K : {K}\n')
            print(f'roc_auc_score:  {area_under_roc}\n')
            print(f'-------------------')

            if best_roc < area_under_roc:
                best_roc = area_under_roc
                best_K = K
                best_answers = answers
                best_scores = scores


        print(f'-------------------')
        print(f'best_K : {best_K}\n')
        print(f'best_roc:  {best_roc}\n')
        print(f'-------------------')

        precision, recall, thresholds = precision_recall_curve(best_answers, best_scores)
        print(f'precision:  {precision}\n')
        print(f'recall:  {recall}\n')
        print(f'thresholds:  {thresholds}\n')

        fig1 = plt.figure()
        average_precision = average_precision_score(best_answers, best_scores)
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        fig1.savefig(os.path.join(args.path, 'precision_recall.png'))


        fpr, tpr, thresholds = roc_curve(best_answers, best_scores)
        area_under_roc = roc_auc_score(best_answers, best_scores)

        fpr_div_tpr = [x/y for x, y in zip(fpr, tpr) if y!=0]
        optimal_threshold = thresholds[np.argmax(fpr_div_tpr)]

        print(f'-------------------')
        print(f'false positive rate:  {fpr}\n')
        print(f'true positive rate:  {tpr}\n')
        print(f'thresholds:  {thresholds}\n')
        print(f'area_under_roc:  {area_under_roc}\n')
        print(f'optimal_threshold:  {optimal_threshold}\n')
        print(f'-------------------')



        ###########################################################
        fig2 = plt.figure()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 0

        # Compute ROC curve and area the curve
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        plt.scatter(fpr, thresholds, color='k', s=1)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic example for K={best_K}')
        plt.legend(loc="lower right")
        # plt.show()
        fig2.savefig(os.path.join(args.path, 'roc.png'))
        print('plot complete')
        ############################################################

    elif args.test_after_tune:

        with torch.no_grad():
            task, it = iters[0]

            # predictions *****
            threshold = args.optimal_th
            scores = []
            predictions = []
            for batch_idx, batch in enumerate(it):
                _, score = model(batch, iteration=1)
                _, seq_len = score.size()
                # score[(score != 1.0).nonzero()
                score_cleaned = torch.where(score < 0.99999, score, torch.zeros_like(score))

                # score_sorted, score_sorted_indices = torch.sort(score, dim=1, descending=True)
                # score = torch.mean(score_sorted[:, int(1/K * seq_len): int((1-1/K) * seq_len)], dim=1)

                score = torch.mean(torch.topk(score_cleaned, args.K, dim=1)[0], dim=1)
                for i, ss in enumerate(score):
                    scores.append(ss)
                    if ss > threshold:
                        pp = "positive"
                    else:
                        pp = "negative"
                    predictions.append(pp)

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
                    answers.append(aa)

            # results ***
            metrics, answers = compute_metrics(predictions, answers,
                                               bleu='iwslt' in task or 'multi30k' in task or 'almond' in task,
                                               dialogue='woz' in task,
                                               rouge='cnn' in task, logical_form='sql' in task, corpus_f1='zre' in task,
                                               func_accuracy='almond' in task and not args.reverse_task_bool,
                                               dev_accuracy='almond' in task and not args.reverse_task_bool,
                                               args=args)

            metric_entry = ''
            for metric_key, metric_value in metrics.items():
                metric_entry += f'{metric_key}_{metric_value:.2f}:'
            metric_entry = metric_entry[:-1]

            print(f'-------------------')
            print(metric_entry)
            print(f'-------------------')


    else:
        decaScore = []
        with torch.no_grad():
            for task, it in iters:
                print(task)
                if args.eval_dir:
                    prediction_file_name = os.path.join(args.eval_dir, os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task + '.txt'))
                    answer_file_name = os.path.join(args.eval_dir, os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task + '.gold.txt'))
                    results_file_name = answer_file_name.replace('gold', 'results')
                else:
                    prediction_file_name = os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task + '.txt')
                    answer_file_name = os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task + '.gold.txt')
                    results_file_name = answer_file_name.replace('gold', 'results')
                if 'sql' in task or 'squad' in task:
                    ids_file_name = answer_file_name.replace('gold', 'ids')
                if os.path.exists(prediction_file_name):
                    print('** ', prediction_file_name, ' already exists -- this is where predictions are stored **')
                if os.path.exists(answer_file_name):
                    print('** ', answer_file_name, ' already exists -- this is where ground truth answers are stored **')
                if os.path.exists(results_file_name):
                    print('** ', results_file_name, ' already exists -- this is where metrics are stored **')
                    with open(results_file_name) as results_file:
                      for l in results_file:
                          print(l)
                    if not args.overwrite_predictions and args.silent:
                        with open(results_file_name) as results_file:
                              metrics = json.loads(results_file.readlines()[0])
                              decaScore.append(metrics[args.task_to_metric[task]])
                        continue

                for x in [prediction_file_name, answer_file_name, results_file_name]:
                    os.makedirs(os.path.dirname(x), exist_ok=True)

                if not os.path.exists(prediction_file_name) or args.overwrite_predictions:
                    print('** overwriting old results with new predictions **')
                    with open(prediction_file_name, 'w') as prediction_file:
                        predictions = []
                        ids = []
                        for batch_idx, batch in enumerate(it):
                            _, p = model(batch, iteration=1)

                            if task == 'almond':
                                setattr(field, 'use_revtok', False)
                                setattr(field, 'tokenize', tokenizer)
                                p = field.reverse_almond(p)
                                setattr(field, 'use_revtok', True)
                                setattr(field, 'tokenize', get_tokenizer('revtok'))
                            else:
                                p = field.reverse(p)

                            for i, pp in enumerate(p):
                                if 'sql' in task:
                                    ids.append(int(batch.wikisql_id[i]))
                                if 'squad' in task:
                                    ids.append(it.dataset.q_ids[int(batch.squad_id[i])])
                                prediction_file.write(json.dumps(pp) + '\n')
                                predictions.append(pp)
                    if 'sql' in task:
                        with open(ids_file_name, 'w') as id_file:
                            for i in ids:
                                id_file.write(json.dumps(i) + '\n')
                    if 'squad' in task:
                        with open(ids_file_name, 'w') as id_file:
                            for i in ids:
                                id_file.write(i + '\n')
                else:
                    with open(prediction_file_name) as prediction_file:
                        predictions = [x.strip() for x in prediction_file.readlines()]
                    if 'sql' in task or 'squad' in task:
                        with open(ids_file_name) as id_file:
                            ids = [int(x.strip()) for x in id_file.readlines()]

                def from_all_answers(an):
                    return [it.dataset.all_answers[sid] for sid in an.tolist()]

                if not os.path.exists(answer_file_name):
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
                                if task == 'almond':
                                    setattr(field, 'use_revtok', False)
                                    setattr(field, 'tokenize', tokenizer)
                                    a = field.reverse_almond(batch.answer.data)
                                    setattr(field, 'use_revtok', True)
                                    setattr(field, 'tokenize', 'revtok')
                                else:
                                    a = field.reverse(batch.answer.data)
                            for aa in a:
                                answers.append(aa)
                                answer_file.write(json.dumps(aa) + '\n')
                else:
                    with open(answer_file_name) as answer_file:
                        answers = [json.loads(x.strip()) for x in answer_file.readlines()]

                if len(answers) > 0:
                    if not os.path.exists(results_file_name) or args.overwrite_predictions:
                        metrics, answers = compute_metrics(predictions, answers,
                                               bleu='iwslt' in task or 'multi30k' in task or 'almond' in task,
                                               dialogue='woz' in task,
                                               rouge='cnn' in task, logical_form='sql' in task, corpus_f1='zre' in task,
                                               func_accuracy='almond' in task and not args.reverse_task_bool,
                                               dev_accuracy='almond' in task and not args.reverse_task_bool,
                                               args=args)
                        with open(results_file_name, 'w') as results_file:
                            results_file.write(json.dumps(metrics) + '\n')
                    else:
                        with open(results_file_name) as results_file:
                            metrics = json.loads(results_file.readlines()[0])

                    if not args.silent:
                        for i, (p, a) in enumerate(zip(predictions, answers)):
                            print(f'Prediction {i+1}: {p}\nAnswer {i+1}: {a}\n')
                    print(metrics)
                    decaScore.append(metrics[args.task_to_metric[task]])
        print(f'Evaluated Tasks:\n')
        for i, (task, _) in enumerate(iters):
            print(f'{task}: {decaScore[i]}')
        print(f'-------------------')
        print(f'DecaScore:  {sum(decaScore)}\n')

        print(f'\nSummary: | {sum(decaScore)} | {" | ".join([str(x) for x in decaScore])} |\n')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--evaluate', type=str, required=True)
    parser.add_argument('--tasks', default=['almond', 'squad', 'iwslt.en.de', 'cnn_dailymail', 'multinli.in.out', 'sst', 'srl', 'zre', 'woz.en', 'wikisql', 'schema'], nargs='+')
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used (multi-gpu currently WIP)')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--data', default='./decaNLP/.data/', type=str, help='where to load data from.')
    parser.add_argument('--embeddings', default='./decaNLP/.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--checkpoint_name')
    parser.add_argument('--bleu', action='store_true', help='whether to use the bleu metric (always on for iwslt)')
    parser.add_argument('--rouge', action='store_true', help='whether to use the bleu metric (always on for cnn, dailymail, and cnn_dailymail)')
    parser.add_argument('--overwrite_predictions', action='store_true', help='whether to overwrite previously written predictions')
    parser.add_argument('--silent', action='store_true', help='whether to print predictions to stdout')

    parser.add_argument('--skip_cache', action='store_true', dest='skip_cache_bool', help='whether to use exisiting cached splits or generate new ones')
    parser.add_argument('--reverse_task', action='store_true', dest='reverse_task_bool', help='whether to translate english to code or the other way around')
    parser.add_argument('--eval_dir', type=str, default=None, help='use this directory to store eval results')

    parser.add_argument('--tune', action='store_true', help='whether to tune or predict')

    parser.add_argument('--test_after_tune', action='store_true', help='test after tuning the model and finding the optimal K and threshold')
    parser.add_argument('--K', default=10, type=int, help='optimal K for topk')
    parser.add_argument('--optimal_th', default=0.5, type=float, help='optimal threshold to gain highest TP/FP')

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
            score = float(l.split(metric+'_')[1].split(':')[0])
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

    # save_dict = torch.load(args.best_checkpoint)
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
