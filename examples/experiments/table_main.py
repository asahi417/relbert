import os
import numpy as np
import pandas as pd
from relbert.util import wget

# get word embedding result
os.makedirs('./cache', exist_ok=True)
wget('https://raw.githubusercontent.com/asahi417/AnalogyTools/master/results/analogy_test.csv', './cache')
path_analogy_result = './cache/analogy_test.csv'
wget('https://raw.githubusercontent.com/asahi417/AnalogyTools/master/results/lexical_relation.csv', './cache')
path_lex_result = './cache/lexical_relation.csv'
df_we = pd.read_csv(path_lex_result, index_col=0)


# Analogy result
os.makedirs('./relbert_output/eval/latex_table', exist_ok=True)
df = pd.read_csv('./relbert_output/eval/analogy.csv', index_col=0)

df = df.sort_values(by=['validation_loss', 'data'])
df_vanilla = df[df.template_type == df.template_type]
df = df[df.template_type != df.template_type]

best_models = {}
for lm in ['roberta', 'bert', 'albert']:
    df_tmp = df[[lm == i.split('/')[-2].split('_')[0] for i in df.model]]
    df_tmp_v = df_vanilla[[lm == i.split('-')[0] for i in df_vanilla.model]]
    cat = []
    cat_v = []
    best_models[lm] = {}
    for method in ['custom', 'auto_c', 'auto_d']:
        if method != 'custom' and lm != 'roberta':
            continue
        df_tmp_tmp = df_tmp[[method in i for i in df_tmp.model]]
        tmp = df_tmp_tmp.head(5)[['accuracy/test', 'accuracy/full']] * 100
        tmp.columns = [method, 'accuracy_full']
        tmp.index = df_tmp_tmp.head(5)['data']
        sat_full = tmp['accuracy_full'].T['sat']
        tmp = tmp[method]
        tmp['sat_full'] = sat_full
        cat.append(tmp)
        best_models[lm][method] = df_tmp_tmp.head(1).model.values[0].replace('./', '')

        # get vanilla LM result
        if lm != 'roberta':
            continue
        template_type = df_tmp_tmp['model'].head(1).values[0].split('/')[-2].split('_')[-1]
        df_tmp_tmp_v = df_tmp_v[[template_type in i for i in df_tmp_v.template_type]]
        tmp = df_tmp_tmp_v.head(5)[['accuracy/test', 'accuracy/full']] * 100
        tmp.columns = [method, 'accuracy_full']
        tmp.index = df_tmp_tmp_v.head(5)['data']
        sat_full = tmp['accuracy_full'].T['sat']
        tmp = tmp[method]
        tmp['sat_full'] = sat_full
        cat_v.append(tmp)

    df_out = pd.concat(cat, axis=1).T.round(1)
    df_out.to_csv('./relbert_output/eval/summary/analogy.relbert.{}.csv'.format(lm))
    if len(cat_v):
        df_out_v = pd.concat(cat_v, axis=1).T.round(1)
        # print(df_out_v.to_latex())
        # input()
        df_out_v.to_csv('./relbert_output/eval/summary/analogy.vanilla.{}.csv'.format(lm))
print(best_models)
# lexical relation classification
df = pd.read_csv('./relbert_output/eval/relation_classification.csv', index_col=0)
df = df.sort_values(by=['data'])
dataset = ['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']
lm = 'roberta'
f1 = []
for method in ['custom', 'auto_d', 'auto_c']:
    df_tmp = df[df.model == best_models[lm][method]]
    tmp = df_tmp[['f1_macro/test', 'f1_micro/test']].round(3) * 100
    tmp.index = df_tmp.data
    f1 += [np.concatenate([tmp.T[i].values for i in dataset])]
f1 = [[' ', x] + i for x, i in zip(['Manual', 'AutoPrompt', 'P-tuning'], np.array(f1).tolist())]
df = pd.DataFrame(f1, columns=['tmp'] * 2 + ['macro', 'micro'] * 5)
tmp = df.to_latex(index=False)
tmp = tmp.replace('  &     Manual', r"\multirow{3}{*}{\rotatebox{90}{RelBERT}} &     Manual")
tmp = tmp.replace(r'tmp &        tmp &',
r'\multicolumn{2}{c}{\multirow{2}{*}{\textbf{Model}}} ' \
r'& \multicolumn{2}{c}{\textbf{BLESS}} ' \
r'& \multicolumn{2}{c}{\textbf{CogALexV}} ' \
r'& \multicolumn{2}{c}{\textbf{EVALution}} ' \
r'& \multicolumn{2}{c}{\textbf{K\&H+N}} ' \
r'& \multicolumn{2}{c}{\textbf{ROOT09}} \\' + '\n  &  &')
print('\n******* main table *******\n')
print(tmp)
print()
