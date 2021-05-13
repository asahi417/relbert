import os
from itertools import chain
import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt

# get the best config in terms of loss realization
df = pd.read_csv('./asset/analogy.csv', index_col=0)
df_vanilla = df[df.template_type == df.template_type].sort_values(by=['validation_loss', 'data'])
df = df[df.template_type != df.template_type].sort_values(by=['validation_loss', 'data'])
cat = []
index = []
for lm in ['roberta', 'bert', 'albert']:
    df_tmp = df[[lm == i.split('/')[-2].split('_')[0] for i in df.model]]
    df_tmp_tmp = df_tmp[['custom' in i for i in df_tmp.model]]
    tmp = df_tmp_tmp.head(5)[['accuracy/test', 'accuracy/full']] * 100
    tmp.columns = [lm, 'accuracy_full']
    tmp.index = df_tmp_tmp.head(5)['data'].to_list()
    sat_full = tmp['accuracy_full'].T['sat']
    tmp = tmp[lm]
    tmp['sat_full'] = sat_full
    cat.append(tmp)
    if lm != 'roberta':
        continue

    df_tmp_tmp = df_vanilla[[lm in i.split('-')[0] for i in df_vanilla.model]]
    tmp = df_tmp_tmp.head(5)[['accuracy/test', 'accuracy/full']] * 100
    tmp.columns = [lm + '/vanilla', 'accuracy_full']
    tmp.index = df_tmp_tmp.head(5)['data'].to_list()
    sat_full = tmp['accuracy_full'].T['sat']
    tmp = tmp[lm + '/vanilla']
    tmp['sat_full'] = sat_full
    cat.append(tmp)
index = list(chain(*[[i.name] * len(i) for i in cat]))
accuracy = list(chain(*[i.values.tolist() for i in cat]))
data = list(chain(*[i.index.tolist() for i in cat]))
df = pd.DataFrame([index, accuracy, data], index=['Model', 'Accuracy', 'Data']).T
df['Data'] = [i.replace('bats', 'BATS').replace('u2', 'U2').replace('u4', 'U4').
              replace('google', 'Google').replace('sat', 'SAT') for i in df.Data]
df['Model'] = [i.replace('albert', 'ALBERT').replace('roberta', 'RoBERTa').replace('/', ' (vanilla)').
               replace('bert', 'BERT') for i in df.Model]

# plot
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
sns.set_theme(style="darkgrid")
root_dir = './asset/figure'
os.makedirs(root_dir, exist_ok=True)
fontsize = 20

fig = plt.figure()
fig.clear()
ax = sns.barplot(data=df, x='Data', y='Accuracy', hue='Model', order=['SAT', 'U2', 'U4', 'BATS', 'Google'],
                 hue_order=['ALBERT', 'BERT', 'RoBERTa', 'RoBERTa (vanilla)'])
ax.set(ylim=(0, 100))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.setp(ax.get_legend().get_texts(), fontsize=fontsize)
ax.set_xlabel(None)
ax.set_ylabel('Accuracy', fontsize=fontsize)
ax.tick_params(labelsize=fontsize)
fig = ax.get_figure()
plt.tight_layout()
fig.savefig('{}/fig.lm_comparison.png'.format(root_dir))
plt.close()
#
#
# #####################
# df = pd.read_csv('./relbert_output/eval/analogy.csv', index_col=0)
# df = df.sort_values(by=['validation_loss', 'data'])
# df_vanilla = df[df.template_type == df.template_type]
# df = df[df.template_type != df.template_type]
# main_tmp = []
# for lm in ['roberta', 'bert', 'albert']:
#     df_tmp = df[[lm == i.split('/')[-2].split('_')[0] for i in df.model]]
#     df_tmp_v = df_vanilla[[lm == i.split('-')[0] for i in df_vanilla.model]]
#     cat, cat_v = [], []
#     for method in ['custom', 'auto_d', 'auto_c']:
#         if method != 'custom' and lm != 'roberta':
#             continue
#         df_tmp_tmp = df_tmp[[method in i for i in df_tmp.model]]
#         tmp = df_tmp_tmp.head(5)[['accuracy/test', 'accuracy/full']] * 100
#         tmp.columns = [method, 'accuracy_full']
#         tmp.index = df_tmp_tmp.head(5)['data'].to_list()
#         sat_full = tmp['accuracy_full'].T['sat']
#         tmp = tmp[method]
#         tmp['sat_full'] = sat_full
#         cat.append(tmp)
#
#         # get vanilla LM result
#         if lm != 'roberta':
#             continue
#         template_type = df_tmp_tmp['model'].head(1).values[0].split('/')[-2].split('_')[-1]
#         df_tmp_tmp_v = df_tmp_v[[template_type in i for i in df_tmp_v.template_type]]
#         tmp = df_tmp_tmp_v.head(5)[['accuracy/test', 'accuracy/full']] * 100
#         tmp.columns = [method, 'accuracy_full']
#         tmp.index = df_tmp_tmp_v.head(5)['data'].to_list()
#         sat_full = tmp['accuracy_full'].T['sat']
#         tmp = tmp[method]
#         tmp['sat_full'] = sat_full
#         cat_v.append(tmp)
#
#     df_out = pd.concat(cat, axis=1).T.round(1)[['sat_full', 'sat', 'u2', 'u4', 'google', 'bats']]
#     df_out.index = ['Manual', 'AutoPrompt', 'P-tuning']
#     print(df_out)
#     # df_out.to_csv('./relbert_output/eval/summary/analogy.relbert.{}.csv'.format(lm))
#     if len(cat_v):
#         df_out_v = pd.concat(cat_v, axis=1).T.round(1)
#         # print(df_out_v.to_latex())
#         # input()
#         # df_out_v.to_csv('./relbert_output/eval/summary/analogy.vanilla.{}.csv'.format(lm))
