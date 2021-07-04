""" Ablation study: Train model without hypernym """
import os
import logging
from copy import deepcopy
from itertools import chain

import pandas as pd
import relbert
from relbert.evaluator import evaluate_classification


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
path = 'asset/accuracy.classification.exclusion_test.csv'
if not os.path.exists(path):
    export = 'relbert_output/ablation_study/exclusion_test/ckpt'
    os.makedirs(os.path.dirname(export), exist_ok=True)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    file_handler = logging.FileHandler('relbert_output/ablation_study/exclusion_test/log.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    logger.addHandler(file_handler)

    ##########################
    # Model without Hypernym #
    ##########################
    if not os.path.exists(export):
        trainer = relbert.Trainer(
            model='roberta-large',
            template_type='c',
            epoch=2,
            export=export,
            exclude_relation="Class Inclusion"
        )
        trainer.train()

    full_result = []
    target_relation = list(chain(*list(shared_relation.values())))
    # load checkpoint from model hub
    full_result += evaluate_classification(relbert_ckpt="asahi417/relbert_roberta_custom_c",
                                           target_relation=target_relation)
    full_result += evaluate_classification(relbert_ckpt='{}/epoch_2'.format(export),
                                           target_relation=target_relation)

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
    os.makedirs(os.path.dirname(export), exist_ok=True)
    pd.DataFrame(full_result_new).to_csv(export)

df = pd.read_csv(path, index_col=0)
df = df[[c for c in df.columns if 'f1' in c or c in ['model', 'data']]]
df_new = df[['model', 'data', 'f1_macro/test', 'f1_micro/test']].copy()
for k, v in shared_relation.items():
    tmp = 0

    for _v in v:
        if _v == 'PartOf':
            tmp += df['f1/test/{}'.format(_v)].fillna(0).to_numpy() * 145 / (145 + 86)
        elif _v == 'MadeOf':
            tmp += df['f1/test/{}'.format(_v)].fillna(0).to_numpy() * 86 / (145 + 86)
        else:
            tmp += df['f1/test/{}'.format(_v)].fillna(0).to_numpy()
    df_new[k] = tmp
df_new.index = df_new.pop('data').tolist()
model_ex = [i for i in df.model.unique() if 'exclusion' in i][0]
model = [i for i in df.model.unique() if 'exclusion' not in i][0]
df_new['macro'] = df_new.pop('f1_macro/test')
df_new['micro'] = df_new.pop('f1_micro/test')
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
