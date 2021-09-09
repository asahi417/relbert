""" Plot figure 3 """
import os
from itertools import chain
import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt

# get the best config in terms of loss realization
df = pd.read_csv('examples/experiments/output/eval/accuracy.analogy.csv', index_col=0)
df_vanilla = df[df.template_type == df.template_type].sort_values(by=['validation_loss', 'data'])
df = df[df.template_type != df.template_type].sort_values(by=['validation_loss', 'data'])
cat = []
index = []
method = 'custom'
for lm in ['roberta', 'bert', 'albert']:
    df_tmp = df[[lm == i.split('/')[-2].split('_')[0] for i in df.model]]
    df_tmp_tmp = df_tmp[[method in i for i in df_tmp.model]]
    tmp = df_tmp_tmp.head(5)[['accuracy/test', 'accuracy/full']] * 100
    tmp.columns = [lm, 'accuracy_full']
    tmp.index = df_tmp_tmp.head(5)['data'].to_list()
    sat_full = tmp['accuracy_full'].T['sat']
    tmp = tmp[lm]
    tmp['sat_full'] = sat_full
    cat.append(tmp)
    if lm != 'roberta':
        continue

    template = df_tmp_tmp.head(1).model.values[0]
    if method == 'custom':
        template = template.split('custom_')[1][0]
    else:
        template = './' + template.replace('ckpt', 'prompt_files').replace('auto_', '').replace('epoch_1', 'prompt.json')

    df_tmp_tmp = df_vanilla[[lm in i.split('-')[0] for i in df_vanilla.model]]
    df_tmp_tmp = df_tmp_tmp[df_tmp_tmp.template_type == template]
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
df['Model'] = [i.replace('albert', 'ALBERT').replace('roberta', 'RoBERTa').replace('/vanilla', ' (vanilla)').
               replace('bert', 'BERT') for i in df.Model]

# plot
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
# sns.set_theme(style="darkgrid")
root_dir = 'examples/experiments/output/figure'
os.makedirs(root_dir, exist_ok=True)
fontsize = 18

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

################################################
# get the best config in terms of loss realization
df = pd.read_csv('examples/experiments/output/eval/accuracy.analogy.csv', index_col=0)
df_vanilla = df[df.template_type == df.template_type].sort_values(by=['validation_loss', 'data'])
df = df[df.template_type != df.template_type].sort_values(by=['validation_loss', 'data'])
cat = []
index = []
lm = 'roberta'
df_tmp = df[[lm == i.split('/')[-2].split('_')[0] for i in df.model]]
for method in ['custom', 'auto_c', 'auto_d']:
    df_tmp_tmp = df_tmp[[method in i for i in df_tmp.model]]
    tmp = df_tmp_tmp.head(5)[['accuracy/test', 'accuracy/full']] * 100
    tmp.columns = [method, 'accuracy_full']
    tmp.index = df_tmp_tmp.head(5)['data'].to_list()
    sat_full = tmp['accuracy_full'].T['sat']
    tmp = tmp[method]
    tmp['sat_full'] = sat_full
    cat.append(tmp)

    template = df_tmp_tmp.head(1).model.values[0]
    if method == 'custom':
        template = template.split('custom_')[1][0]
    else:
        template = './' + template.replace('ckpt', 'prompt_files').replace('auto_', '').replace('epoch_1',
                                                                                                'prompt.json')

    df_tmp_tmp = df_vanilla[[lm in i.split('-')[0] for i in df_vanilla.model]]
    df_tmp_tmp = df_tmp_tmp[df_tmp_tmp.template_type == template]
    tmp = df_tmp_tmp.head(5)[['accuracy/test', 'accuracy/full']] * 100
    tmp.columns = [method + '/vanilla', 'accuracy_full']
    tmp.index = df_tmp_tmp.head(5)['data'].to_list()
    # print(tmp, method)
    sat_full = tmp['accuracy_full'].T['sat']
    tmp = tmp[method + '/vanilla']
    tmp['sat_full'] = sat_full
    cat.append(tmp)
index = list(chain(*[[i.name] * len(i) for i in cat]))
accuracy = list(chain(*[i.values.tolist() for i in cat]))
data = list(chain(*[i.index.tolist() for i in cat]))
df = pd.DataFrame([index, accuracy, data], index=['Model', 'Accuracy', 'Data']).T
df['Data'] = [i.replace('bats', 'BATS').replace('u2', 'U2').replace('u4', 'U4').
                  replace('google', 'Google').replace('sat', 'SAT') for i in df.Data]
df['Model'] = [i.replace('custom', 'Manual').replace('auto_c', 'P-tuning').replace('/vanilla', ' (vanilla)').
                   replace('auto_d', 'AutoPrompt') for i in df.Model]

# plot
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
# sns.set_theme(style="darkgrid")
root_dir = 'examples/experiments/output/figure'
os.makedirs(root_dir, exist_ok=True)
fontsize = 15

list_tmp = []
for data_name in ['BATS', 'Google', 'SAT', 'U2', 'U4']:
    _df = df[df.Data == data_name]
    for m in ['Manual', 'AutoPrompt', 'P-tuning']:
        a = _df[_df.Model == m]['Accuracy'].values[0] - _df[_df.Model == m + ' (vanilla)']['Accuracy'].values[0]
        list_tmp.append([m, a, data_name])
df_tmp = pd.DataFrame(list_tmp, columns=['Model', 'Accuracy', 'Data'])
df = df_tmp
fig = plt.figure()
fig.clear()
# order = ['Manual', 'Manual (vanilla)',
#          'AutoPrompt', 'AutoPrompt (vanilla)',
#          'P-tuning', 'P-tuning (vanilla)']
order = ['Manual', 'AutoPrompt', 'P-tuning']
ax = sns.barplot(data=df, x='Data', y='Accuracy', hue='Model', order=['SAT', 'U2', 'U4', 'BATS', 'Google'],
                 hue_order=order)
# ax.set(ylim=(0, 60))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.setp(ax.get_legend().get_texts(), fontsize=10)
ax.set_xlabel(None)
ax.set_ylabel('Accuracy Drop', fontsize=fontsize)
ax.tick_params(labelsize=fontsize)
fig = ax.get_figure()
plt.tight_layout()
fig.savefig('{}/fig.finetuning.png'.format(root_dir))
plt.close()
