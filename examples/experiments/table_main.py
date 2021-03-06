""" Export Latex Table for main results: Table 1 & 2 (Analogy and relation classification accuracy) """
import json
import os
from itertools import chain
import numpy as np
import pandas as pd
from relbert.util import wget
import warnings
warnings.filterwarnings("ignore")

# get word embedding result
os.makedirs('./cache', exist_ok=True)

##########################################
# format word embedding result (analogy) #
##########################################
wget('https://raw.githubusercontent.com/asahi417/AnalogyTools/master/results/analogy_test.csv', './cache')
path_analogy_result = './cache/analogy_test.csv'
df = pd.read_csv(path_analogy_result, index_col=0)
ttmp = []
for i in ['glove', 'fasttext']:
    tmp = df[df.model == i][df.feature == 'diff'][df.add_relative == False][df.add_pair2vec == False]
    tmp.index = tmp.data.tolist()
    sat_full = tmp.loc['sat']['accuracy']
    tmp = tmp[['accuracy_test']].T
    tmp['sat_full'] = sat_full
    tmp.index = [i.replace('fasttext', 'FastText').replace('glove', 'GloVe')]
    ttmp.append(tmp)
tmp = df[df.only_pair_embedding == True][df.add_relative == True]
tmp.index = tmp.data.tolist()
sat_full = tmp.loc['sat']['accuracy']
tmp = tmp[['accuracy_test']].T
tmp['sat_full'] = sat_full
tmp.index = ['RELATIVE']
ttmp.append(tmp)
tmp = df[df.only_pair_embedding == True][df.add_relative == False]
tmp.index = tmp.data.tolist()
sat_full = tmp.loc['sat']['accuracy']
tmp = tmp[['accuracy_test']].T
tmp['sat_full'] = sat_full
tmp.index = ['pair2vec']
ttmp.append(tmp)
df_analogy_we = pd.concat(ttmp)[['sat_full', 'sat', 'u2', 'u4', 'google', 'bats']].round(3) * 100

#################################################
# format word embedding result (classification) #
#################################################
wget('https://raw.githubusercontent.com/asahi417/AnalogyTools/master/results/lexical_relation.csv', './cache')
path_lex_result = './cache/lexical_relation.csv'
df = pd.read_csv(path_lex_result, index_col=0)
dataset = ['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']
ttmp = []
for i in ['glove', 'fasttext']:
    for n, p in zip(['cat', 'cat+dot', 'diff', 'diff+dot'],
                    ['concat', "('concat', 'dot')", 'diff', "('diff', 'dot')"]):
        tmp = df[df.feature == p][df.model == i]
        _tmp = tmp[tmp.add_pair2vec == False][tmp.add_relative == False]
        _tmp.index = _tmp.data.tolist()
        _tmp = (_tmp[['metric/test/f1_macro', 'metric/test/f1_micro']] * 100).round(1)
        data = list(chain(*[_tmp.T[i].tolist() for i in dataset]))
        ttmp.append([i, n] + data)
        if n not in ['cat', 'diff']:
            _tmp = tmp[tmp.add_pair2vec == True]
            _tmp.index = _tmp.data.tolist()
            _tmp = (_tmp[['metric/test/f1_macro', 'metric/test/f1_micro']] * 100).round(1)
            data = list(chain(*[_tmp.T[i].tolist() for i in dataset]))
            ttmp.append([i, n+'+pair'] + data)
            _tmp = tmp[tmp.add_relative == True]
            _tmp.index = _tmp.data.tolist()
            _tmp = (_tmp[['metric/test/f1_macro', 'metric/test/f1_micro']] * 100).round(1)
            data = list(chain(*[_tmp.T[i].tolist() for i in dataset]))
            ttmp.append([i, n + '+rel'] + data)
df_classification_we = pd.DataFrame(ttmp)
df_classification_we.index = df_classification_we.pop(0).tolist()
df_classification_we.columns = [''] + ['macro', 'micro'] * 5

#####################
# RelBERT (analogy) #
#####################
df = pd.read_csv('examples/experiments/output/eval/accuracy.analogy.csv', index_col=0)
df = df[df.template_type != df.template_type].sort_values(by=['validation_loss', 'data'])
lm = 'roberta'
# lm = 'albert'
# lm = 'bert'
df_tmp = df[[lm == i.split('/')[-2].split('_')[0] for i in df.model]]
cat = []
best_models = {}
for method in ['custom', 'auto_d', 'auto_c']:
    df_tmp_tmp = df_tmp[[method in i for i in df_tmp.model]]
    tmp = df_tmp_tmp.head(5)[['accuracy/test', 'accuracy/full']] * 100
    tmp.columns = [method, 'accuracy_full']
    tmp.index = df_tmp_tmp.head(5)['data'].to_list()
    sat_full = tmp['accuracy_full'].T['sat']
    tmp = tmp[method]
    tmp['sat_full'] = sat_full
    cat.append(tmp)
    best_models[method] = df_tmp_tmp.head(1).model.values[0].replace('./', '')
df_out = pd.concat(cat, axis=1).T.round(1)[['sat_full', 'sat', 'u2', 'u4', 'google', 'bats']]
df_out.index = [r'$\cdot$ Manual', r'$\cdot$ AutoPrompt', r'$\cdot$ P-tuning']
df_analogy = pd.concat([df_analogy_we, df_out]).round(1)
print(df_analogy.to_markdown())
# input()

print('\n******* config [ANALOGY] *******\n')
print(best_models)


############################
# RelBERT (classification) #
############################
df = pd.read_csv('examples/experiments/output/eval/accuracy.classification.csv', index_col=0)
df = df.sort_values(by=['data'])
f1 = []
config = {}
valid = 'val/f1_macro'
for method in ['custom', 'auto_d', 'auto_c']:
    config[method] = {}
    df_tmp = df[df.model == best_models[method]]
    df_tmp.index = df_tmp.data
    _f1 = []
    for i in dataset:
        if i == 'CogALexV':
            tmp = df_tmp.T[['CogALexV']].T
        else:
            tmp = df_tmp.T[i].T.sort_values(by=[valid], ascending=False).head(1)
        config[method][i] = tmp['classifier_config'].values[0]
        _f1.append((tmp[['test/f1_macro', 'test/f1_micro']].values[0] * 100))
    f1.append(np.concatenate(_f1))

f1 = [['RelBERT', x] + i for x, i in zip(['Manual', 'AutoPrompt', 'P-tuning'], np.array(f1).tolist())]
df = pd.DataFrame(f1, columns=['tmp', ''] + ['macro', 'micro'] * 5)
df.index = df.pop('tmp')
df_classification = pd.concat([df_classification_we, df])
df_classification.index.name = ''
df_classification = df_classification.round(1)

print('\n******* config [CLASSIFICATION] *******\n')
for method in ['custom', 'auto_d', 'auto_c']:
    print(method)
    print(json.dumps(config[method], indent=4, sort_keys=True))
    print()
# print(tmp)
# target_key = ['hidden_layer_sizes', 'learning_rate_init']
# tmp = {k: {_k: json.loads(v)[_k] for _k in target_key} for k, v in tmp.items()}

####################
# Formatting latex #
####################


def clean_latex(string):
    return string.replace(r'\textbackslash ', '\\').replace(r'\{', '{').replace(r'\}', '}').replace(r'\$', r'$')


print('\n******* main table [ANALOGY] *******\n')
# analogy
df_analogy.columns = [r'\textbf{SAT\dag}',
                      r'\textbf{SAT}',
                      r'\textbf{U2}',
                      r'\textbf{U4}',
                      r'\textbf{Google}',
                      r'\textbf{BATS}']
df_analogy.index.name = r'\textbf{Model}'
table = df_analogy.to_latex()
block = table.split('\n')
table = clean_latex('\n'.join(block[:9] + [r'\midrule \\', r'\multicolumn{7}{l}{RelBERT} \\'] + block[9:]))
print(table)
print()

print('\n******* main table [CLASSIFICATION] *******\n')
# classification
df_classification.index = [' '] * len(df_classification.index)
df_classification[''] = [r'\textit{' + i + '}' if i not in ['Manual', 'AutoPrompt', 'P-tuning'] else i for i in df_classification['']]
block = df_classification.to_latex().split('\n')
header = r'\multicolumn{2}{c}{\multirow{2}{*}{\textbf{Model}}} & \multicolumn{2}{c}{\textbf{BLESS}} ' \
         r'& \multicolumn{2}{c}{\textbf{CogALexV}} & \multicolumn{2}{c}{\textbf{EVALution}} ' \
         r'& \multicolumn{2}{c}{\textbf{K\&H+N}} & \multicolumn{2}{c}{\textbf{ROOT09}} \\'

table = clean_latex('\n'.join(
    [block[0], block[1], header, '{} & ' + block[2], block[3]] +
    [r'\multirow{8}{*}{GloVe}' + block[4]] + block[5:12] +
    [r'\midrule', r'\multirow{8}{*}{FastText}' + block[12]] + block[13:20] +
    [r'\midrule', r"\multirow{3}{*}{RelBERT}" + block[20]] + block[21:]))

print(table)
print()
