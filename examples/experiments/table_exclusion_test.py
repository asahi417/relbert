""" Export Latex Table for main results: Table 3 (overlap analysis) """
import os
import logging
from copy import deepcopy
from itertools import chain

import pandas as pd
import relbert
from relbert.evaluator import evaluate_classification

import warnings
warnings.filterwarnings("ignore")

# anchor model checkpoint
ckpt = 'asahi417/relbert-roberta-large'
# new model
epoch = 1
template_type = 'd'
model = 'roberta-large'
# best configuration among the main experiment


config = {
    'BLESS': {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': 200, 'learning_rate': 'constant', 'learning_rate_init': 1e-05, 'max_fun': 15000, 'max_iter': 200, 'momentum': 0.9, 'n_iter_no_change': 10, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 0, 'shuffle': True, 'solver': 'adam', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False},
    'CogALexV': {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 200, 'momentum': 0.9, 'n_iter_no_change': 10, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 0, 'shuffle': True, 'solver': 'adam', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False},
    'EVALution': {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': 150, 'learning_rate': 'constant', 'learning_rate_init': 0.0001, 'max_fun': 15000, 'max_iter': 200, 'momentum': 0.9, 'n_iter_no_change': 10, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 0, 'shuffle': True, 'solver': 'adam', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False},
    'K&H+N': {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': 200, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 200, 'momentum': 0.9, 'n_iter_no_change': 10, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 0, 'shuffle': True, 'solver': 'adam', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False},
    'ROOT09': {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 1e-05, 'max_fun': 15000, 'max_iter': 200, 'momentum': 0.9, 'n_iter_no_change': 10, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 0, 'shuffle': True, 'solver': 'adam', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}
}
path = 'examples/experiments/output/eval/accuracy.classification.exclusion_test.csv'
export = './examples/experiments/output/ablation_study/exclusion_test/ckpt'


def clean_latex(string):
    return string.replace(r'\textbackslash ', '\\').replace(r'\{', '{').replace(r'\}', '}').replace(r'\$', r'$')


shared_relation = {
    'rand': ['random', 'RANDOM', 'false'],
    'mero': ['PartOf', 'PART_OF', 'mero', 'MadeOf'],
    'event': ['event'],
    'hyp': ['HYPER', 'hyper', 'hypo', 'IsA'],
    'cohyp': ['COORD', 'coord', 'sibl'],
    'attr': ['attri', 'HasProperty'],
    'subs': ['HasA'],
    'ant': ['ANT', 'Antonym'],
    'syn': ['SYN', 'Synonym']
}
dataset = ['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']
if not os.path.exists(path):
    os.makedirs(os.path.dirname(export), exist_ok=True)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    file_handler = logging.FileHandler('examples/experiments/output/ablation_study/exclusion_test/log.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    logger.addHandler(file_handler)

    ##########################
    # Model without Hypernym #
    ##########################
    if not os.path.exists(export):
        trainer = relbert.Trainer(
            model=model,
            template_type=template_type,
            export=export,
            epoch=epoch,
            exclude_relation="Class Inclusion",
            batch=64,
            parent_contrast=True,
            softmax_loss=True,
            in_batch_negative=True
        )
        trainer.train()

    full_result = []
    target_relation = list(chain(*list(shared_relation.values())))
    # load checkpoint from model hub
    full_result += evaluate_classification(relbert_ckpt=ckpt,
                                           target_relation=target_relation,
                                           config=config)
    full_result += evaluate_classification(relbert_ckpt='{}/epoch_{}'.format(export, epoch),
                                           target_relation=target_relation,
                                           config=config)

    full_result_new = []
    for x in full_result:
        i = deepcopy(x)
        for k in i.keys():
            if 'accuracy/test/' in k:
                _k = k.replace('accuracy/test/', '')
                for __k, __v in shared_relation.items():
                    if _k in __v:
                        x[k.replace(_k, __k)] = x.pop(k)
        full_result_new.append(x)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(full_result_new).to_csv(path)

df = pd.read_csv(path, index_col=0)
df = df[[c for c in df.columns if 'f1' in c or c in ['model', 'data']]]
df_new = df[['model', 'data', 'test/f1_macro', 'test/f1_micro']].copy()
for k, v in shared_relation.items():
    tmp = 0

    for _v in v:
        if _v == 'PartOf':
            tmp += df['test/f1/{}'.format(_v)].fillna(0).to_numpy() * 145 / (145 + 86)
        elif _v == 'MadeOf':
            tmp += df['test/f1/{}'.format(_v)].fillna(0).to_numpy() * 86 / (145 + 86)
        else:
            tmp += df['test/f1/{}'.format(_v)].fillna(0).to_numpy()
    df_new[k] = tmp
df_new.index = df_new.pop('data').tolist()
model_ex = [i for i in df.model.unique() if 'exclusion' in i][0]
model = [i for i in df.model.unique() if 'exclusion' not in i][0]
df_new['macro'] = df_new.pop('test/f1_macro')
df_new['micro'] = df_new.pop('test/f1_micro')
df_new_ex = df_new[df_new.model == model_ex]
df_new_ex.pop('model')
df_new_ex = (df_new_ex*100).round(1)
df_new_ex = df_new_ex.T[dataset]

df_new = df_new[df_new.model == model]
df_new.pop('model')
df_new = (df_new*100).round(1)
df_new = df_new.T[dataset]


df_diff = df_new_ex - df_new
df_new_ex = df_new_ex.astype(str).applymap(lambda y: '-' if y == '0.0' else y)
df_diff = df_diff.round(1).astype(str).applymap(lambda y: r'\,(+{})'.format(y) if '-' not in y else r'\,({})'.format(y))
df_diff = df_diff.applymap(lambda y: '' if y == r'\,(+0.0)' else y)
df_done = df_new_ex + df_diff
df_done = df_done.applymap(lambda y: y + r'\,(+0.0)' if y != '-' and '(' not in y else y)
df_done.columns = [r'\textbf{' + i + r'}' for i in df_done.columns]
table = df_done.to_latex()
block = table.split('\n')
table = clean_latex('\n'.join(block[:13] + [r'\midrule'] + block[13:]))
print('\n******* table [EXCLUSION TEST] *******\n')
print(table)
