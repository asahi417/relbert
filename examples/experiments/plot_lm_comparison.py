import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt

plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
sns.set_theme(style="darkgrid")

df = []
for i in ['bert', 'albert', 'roberta']:
    main_df = pd.read_csv('./relbert_output/eval/summary/analogy.relbert.{}.csv'.format(i), index_col=0).T
    main_df['data'] = main_df.index
    main_df['lm'] = i.replace('roberta', 'RoBERTa').replace('albert', 'ALBERT').replace('bert', 'BERT')
    df.append(main_df[['data', 'custom', 'lm']])
df = pd.concat(df)

fontsize = 20
fig = plt.figure()
fig.clear()
df['data'] = [i.replace('bats', 'BATS').replace('u2', 'U2').replace('u4', 'U4').replace('google', 'Google').replace('sat', 'SAT') for i in df.data]
df['accuracy'] = df['custom'].astype(float)
ax = sns.barplot(data=df, x='data', y='accuracy', hue='lm', order=['SAT', 'U2', 'U4', 'BATS', 'Google'], hue_order=['ALBERT', 'BERT', 'RoBERTa'])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.setp(ax.get_legend().get_texts(), fontsize=fontsize)
ax.set_xlabel(None)
ax.set_ylabel('Accuracy', fontsize=fontsize)
ax.tick_params(labelsize=fontsize)
fig = ax.get_figure()
plt.tight_layout()
fig.savefig('./relbert_output/eval/summary/fig.lm.comparison.png')
plt.close()
