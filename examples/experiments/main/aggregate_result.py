import pandas as pd

# Analogy result
df = pd.read_csv('./relbert_output/eval/analogy.csv', index_col=0)
df = df.sort_values(by=['valid_loss', 'analogy_data'])

df = df[df.template_type != df.template_type]
df_vanilla = df[df.template_type == df.template_type]

for lm in ['roberta', 'bert', 'albert']:
    df_tmp = df[[lm == i.split('/')[-2].split('_')[0] for i in df.model]]
    for method in ['custom', 'auto_c', 'auto_d']:
        df_tmp_tmp = df_tmp[[method in i for i in df_tmp.model]]
        tmp = df_tmp_tmp.head(5)[['analogy_accuracy_test', 'analogy_accuracy_full']] * 100
        tmp.index = df_tmp_tmp.head(5)['analogy_data']
        print('\n - Model: {}, Method: {}'.format(lm, method))
        print(tmp.round(1))
        print('* unique models: {}'.format(df_tmp_tmp.model.unique()))
        print('* best model: {}'.format(df_tmp_tmp['model'].head(1).values))
