import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt

# get the best config in terms of loss realization
df = pd.read_csv('./asset/analogy.csv', index_col=0)
df_vanilla = df[df.template_type != df.template_type].sort_values(by=['validation_loss', 'data'])
df = df[df.template_type != df.template_type].sort_values(by=['validation_loss', 'data'])
cat = []
index = []
for lm in ['roberta', 'bert', 'albert']:
    df_tmp = df[[lm == i.split('/')[-2].split('_')[0] for i in df.model]]

    if lm != 'roberta' and method != 'custom':
        continue
    df_tmp_tmp = df_tmp[[method in i for i in df_tmp.model]]
    tmp = df_tmp_tmp.head(5)[['accuracy/test', 'accuracy/full']] * 100
    tmp.columns = [method, 'accuracy_full']
    tmp.index = df_tmp_tmp.head(5)['data'].to_list()
    sat_full = tmp['accuracy_full'].T['sat']
    tmp = tmp[method]
    tmp['sat_full'] = sat_full
    cat.append(tmp)
    index.append('{}/{}'.format(lm, method))


df = pd.read_csv('./asset/relation_classification.csv', index_col=0)
df = df.sort_values(by=['data'])
dataset = ['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']
for lm in ['roberta', 'bert', 'albert']:
    for method in ['custom', 'auto_d', 'auto_c']:
        if lm != 'roberta' and method != 'custom':
            continue

        f1 = []
        for method in ['custom', 'auto_d', 'auto_c']:
            df_tmp = df[df.model == best_models[lm][method]]
            tmp = df_tmp[['f1_macro/test', 'f1_micro/test']].round(3) * 100
            tmp.index = df_tmp.data
            f1 += [np.concatenate([tmp.T[i].values for i in dataset])]
df = pd.DataFrame(f1)  #, columns=['tmp'] * 2 + ['macro', 'micro'] * 5)

plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
sns.set_theme(style="darkgrid")

root_dir = './relbert_output/eval/figure'
os.makedirs(root_dir)
main_df_vanilla = pd.read_csv('./relbert_output/eval/summary/analogy.vanilla.roberta.csv', index_col=0).T
main_df_vanilla['data'] = main_df_vanilla.index
main_df_vanilla['type'] = 'Vanilla LM'
main_df_relbert = pd.read_csv('./relbert_output/eval/summary/analogy.relbert.roberta.csv', index_col=0).T
main_df_relbert['data'] = main_df_relbert.index
main_df_relbert['type'] = 'RelBERT'
fontsize = 20
for i in ['custom', 'auto_c', 'auto_d']:
    fig = plt.figure()
    fig.clear()
    df_vanilla = main_df_vanilla[['data', i, 'type']]
    df_relbert = main_df_relbert[['data', i, 'type']]
    df = pd.concat([df_vanilla, df_relbert], axis=0)
    df.columns = ['data', 'accuracy', 'type']
    df['data'] = [i.replace('bats', 'BATS').replace('u2', 'U2').replace('u4', 'U4').
                  replace('google', 'Google').replace('sat', 'SAT') for i in df.data]
    df['Accuracy'] = df.accuracy.astype(float)
    ax = sns.barplot(data=df, x='data', y='Accuracy', hue='type', order=['SAT', 'U2', 'U4', 'BATS', 'Google'],
                     hue_order=['Vanilla LM', 'RelBERT'])
    ax.set(ylim=(0, 100))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    plt.setp(ax.get_legend().get_texts(), fontsize=fontsize)
    ax.set_xlabel(None)
    ax.set_ylabel('Accuracy', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig('{}/fig.vanilla.{}.png'.format(root_dir, i))
    plt.close()


#####################
df = pd.read_csv('./relbert_output/eval/analogy.csv', index_col=0)
df = df.sort_values(by=['validation_loss', 'data'])
df_vanilla = df[df.template_type == df.template_type]
df = df[df.template_type != df.template_type]
main_tmp = []
for lm in ['roberta', 'bert', 'albert']:
    df_tmp = df[[lm == i.split('/')[-2].split('_')[0] for i in df.model]]
    df_tmp_v = df_vanilla[[lm == i.split('-')[0] for i in df_vanilla.model]]
    cat, cat_v = [], []
    for method in ['custom', 'auto_d', 'auto_c']:
        if method != 'custom' and lm != 'roberta':
            continue
        df_tmp_tmp = df_tmp[[method in i for i in df_tmp.model]]
        tmp = df_tmp_tmp.head(5)[['accuracy/test', 'accuracy/full']] * 100
        tmp.columns = [method, 'accuracy_full']
        tmp.index = df_tmp_tmp.head(5)['data'].to_list()
        sat_full = tmp['accuracy_full'].T['sat']
        tmp = tmp[method]
        tmp['sat_full'] = sat_full
        cat.append(tmp)

        # get vanilla LM result
        if lm != 'roberta':
            continue
        template_type = df_tmp_tmp['model'].head(1).values[0].split('/')[-2].split('_')[-1]
        df_tmp_tmp_v = df_tmp_v[[template_type in i for i in df_tmp_v.template_type]]
        tmp = df_tmp_tmp_v.head(5)[['accuracy/test', 'accuracy/full']] * 100
        tmp.columns = [method, 'accuracy_full']
        tmp.index = df_tmp_tmp_v.head(5)['data'].to_list()
        sat_full = tmp['accuracy_full'].T['sat']
        tmp = tmp[method]
        tmp['sat_full'] = sat_full
        cat_v.append(tmp)

    df_out = pd.concat(cat, axis=1).T.round(1)[['sat_full', 'sat', 'u2', 'u4', 'google', 'bats']]
    df_out.index = ['Manual', 'AutoPrompt', 'P-tuning']
    print(df_out)
    # df_out.to_csv('./relbert_output/eval/summary/analogy.relbert.{}.csv'.format(lm))
    if len(cat_v):
        df_out_v = pd.concat(cat_v, axis=1).T.round(1)
        # print(df_out_v.to_latex())
        # input()
        # df_out_v.to_csv('./relbert_output/eval/summary/analogy.vanilla.{}.csv'.format(lm))
