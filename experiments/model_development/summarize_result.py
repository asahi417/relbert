import json
import pandas as pd
import os
from distutils.dir_util import copy_tree
from relbert import RelBERT

df = pd.read_csv('relbert_output/eval/accuracy.csv', index_col=0)
df['epoch'] = [int(i.rsplit('/', 1)[-1].replace('epoch_', '')) for i in df['model']]
df['loss_function'] = [i.split('.')[1] for i in df['model']]
df['template'] = [os.path.basename(i.split('.')[0]) for i in df['model']]
df = df[df['epoch'] <= 30]
df = df[df['loss_function'] == 'nce_logout']

for mode in ['average_no_mask', 'average', 'mask']:
    mode_alias = mode.replace('_', '-')
    df_ = df[df['mode'] == mode]
    for temp in ['a', 'b', 'c', 'd', 'e']:
        df__ = df_[df_['template'] == temp]
        best_model = df__.sort_values(by=['validation_loss']).head(1)
        best_valid_loss = best_model['validation_loss'].values[0]
        best_model_ckpt = best_model['model'].values[0]
        with open(f'{best_model_ckpt}/trainer_config.json') as f:
            trainer_config = json.load(f)
        trainer_config['epoch'] = int(best_model['epoch'].values[0])
        trainer_config['data'] = 'relbert/semeval2012_relational_similarity'
        model_relbert = RelBERT(best_model_ckpt)
        new_ckpt = f'{os.path.dirname(best_model_ckpt)}/best_model'
        copy_tree(best_model_ckpt, new_ckpt)
        with open(f'{new_ckpt}/trainer_config.json', 'w') as f:
            json.dump(trainer_config, f)
        with open(f'{new_ckpt}/validation_loss.json', 'w') as f:
            json.dump({
                'validation_loss': best_valid_loss,
                'validation_data': 'relbert/semeval2012_relational_similarity',
                'validation_data/exclude_relation': None
              }, f)
        df_analogy = df__[df__['validation_loss'] == best_valid_loss]
        assert len(df_analogy) == 5, df_analogy.shape
        result = {"distance_function": 'cosine_similarity'}
        for d in ['sat', 'u2', 'u4', 'google', 'bats']:
            result[f'{d}/test'] = df_analogy[df_analogy['data'] == d]['accuracy/valid'].values[0]
            result[f'{d}/valid'] = df_analogy[df_analogy['data'] == d]['accuracy/test'].values[0]
        result['sat_full'] = df_analogy[df_analogy['data'] == 'sat']['accuracy/full'].values[0]
        with open(f'{new_ckpt}/analogy.json', 'w') as f:
            json.dump(result, f)

