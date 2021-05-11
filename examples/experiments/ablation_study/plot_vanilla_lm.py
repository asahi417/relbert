import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt

plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
sns.set_theme(style="darkgrid")

main_df_vanilla = pd.read_csv('./relbert_output/eval/summary/analogy.vanilla.roberta.csv', index_col=0).T
main_df_vanilla['data'] = main_df_vanilla.index
main_df_vanilla['type'] = 'Vanilla LM'
main_df_relbert = pd.read_csv('./relbert_output/eval/summary/analogy.relbert.roberta.csv', index_col=0).T
main_df_relbert['data'] = main_df_relbert.index
main_df_relbert['type'] = 'RelBERT'

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
    plt.setp(ax.get_legend().get_texts(), fontsize=18)
    ax.set_xlabel(None)
    ax.set_ylabel('Accuracy', fontsize=18)
    ax.tick_params(labelsize=18)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig('./relbert_output/eval/summary/fig.vanilla.{}.png'.format(i))
    plt.close()
