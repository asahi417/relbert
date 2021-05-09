import pandas as pd
data = ['sat', 'u2', 'u4', 'google', 'bats']
path = 'relbert_output/eval/analogy.csv'
df = pd.read_csv(path, index_col=0)
df = df.sort_values(by=['valid_loss'])

lm = df[df.model == 'roberta-large']

abs_acc = []
rb_custom = df[['roberta_custom' in i for i in df.model]]
abs_acc.append({i: rb_custom[rb_custom.analogy_data == i]['analogy_accuracy_test'].values[0] * 100 for i in data})
rb_auto = df[['roberta_auto_d' in i for i in df.model]]
abs_acc.append({i: rb_auto[rb_auto.analogy_data == i]['analogy_accuracy_test'].values[0] * 100 for i in data})
rb_ptune = df[['roberta_auto_c' in i for i in df.model]]
abs_acc.append({i: rb_ptune[rb_ptune.analogy_data == i]['analogy_accuracy_test'].values[0] * 100 for i in data})
df_abs = pd.DataFrame(abs_acc, index=['Manual', 'AutoPrompt', 'P-tuning'])
print('\n* RelBERT')
print(df_abs.round(1))
df_abs.round(1).to_csv('examples/experiments/ablation_study/output/analogy.accuracy.relbert.csv')

lm_acc = []
best_tp = rb_custom.head(1)['model'].values[0].split('/')[-2][-1]
tmp = lm[lm.template_type == best_tp]
lm_acc.append({i: tmp[tmp.analogy_data == i]['analogy_accuracy_test'].values[0] * 100 for i in data})

filename = rb_auto.head().model.values[0].split('/')[-2].replace('auto_', '')
best_tp = './relbert_output/prompt_files/{}/prompt.json'.format(filename)
tmp = lm[lm.template_type == best_tp]
lm_acc.append({i: tmp[tmp.analogy_data == i]['analogy_accuracy_test'].values[0] * 100 for i in data})

filename = rb_ptune.head().model.values[0].split('/')[-2].replace('auto_', '')
best_tp = './relbert_output/prompt_files/{}/prompt.json'.format(filename)
tmp = lm[lm.template_type == best_tp]
lm_acc.append({i: tmp[tmp.analogy_data == i]['analogy_accuracy_test'].values[0] * 100 for i in data})
df_lm = pd.DataFrame(lm_acc, index=['Manual', 'AutoPrompt', 'P-tuning'])
print('\n* Vanilla LM')
print(df_lm.round(1))
df_lm.round(1).to_csv('examples/experiments/ablation_study/output/analogy.accuracy.lm.csv')
# df_diff = df_lm - df_abs
# df_lm.round(1).astype(str) +  ' (' + df_diff.round(1).astype(str) + ')'

