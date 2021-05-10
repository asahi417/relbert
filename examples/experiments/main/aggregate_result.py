import os
import pandas as pd

os.makedirs('./relbert_output/eval/summary', exist_ok=True)

# Analogy result
# ',accuracy/valid,accuracy/test,accuracy/full,data,validation_loss,validation_data,model,mode,template_type'
df = pd.read_csv('./relbert_output/eval/analogy.csv', index_col=0)
df = df.sort_values(by=['validation_loss', 'data'])

df_vanilla = df[df.template_type == df.template_type]
df = df[df.template_type != df.template_type]

best_models = {}
for lm in ['roberta']:
# for lm in ['roberta', 'bert', 'albert']:
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
        template_type = df_tmp_tmp['model'].head(1).values[0].split('/')[-2].split('_')[-1]
        df_tmp_tmp_v = df_tmp_v[[template_type in i for i in df_tmp_v.template_type]]
        print(lm, method)

        tmp = df_tmp_tmp_v.head(5)[['accuracy/test', 'accuracy/full']] * 100
        tmp.columns = [method, 'accuracy_full']
        tmp.index = df_tmp_tmp_v.head(5)['data']
        sat_full = tmp['accuracy_full'].T['sat']
        tmp = tmp[method]
        tmp['sat_full'] = sat_full
        cat_v.append(tmp)

    df_out = pd.concat(cat, axis=1).T.round(1)
    df_out.to_csv('./relbert_output/eval/summary/analogy.relbert.{}.csv'.format(lm))
    df_out_v = pd.concat(cat_v, axis=1).T.round(1)
    df_out_v.to_csv('./relbert_output/eval/summary/analogy.vanilla.{}.csv'.format(lm))

print(best_models)
# lexical relation classification
# df = pd.read_csv('./relbert_output/eval/classification.csv', index_col=0)
# df = df.sort_values(by=['data'])
# lm = 'roberta'
# for method in ['custom', 'auto_c', 'auto_d']:
#     df_tmp = df[df.model == best_models[lm][method]]
