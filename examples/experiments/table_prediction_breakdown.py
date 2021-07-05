import json
import os
from itertools import chain
import pandas as pd
import relbert
from relbert.util import wget

wget('https://raw.githubusercontent.com/asahi417/AnalogyTools/master/results/analogy.prediction.json', './cache')
with open('./cache/analogy.prediction.json') as f:
    we_predictions = json.load(f)

path_relbert_pred = './relbert_output/prediction/prediction.relbert.json'
models = ["relbert_output/ckpt/roberta_custom_d/epoch_1",
          "relbert_output/ckpt/roberta_auto_d933/epoch_1",
          "relbert_output/ckpt/roberta_auto_c923/epoch_1"]


def clean_latex(string):
    return string.replace(r'\textbackslash ', '\\').replace(r'\{', '{').replace(r'\}', '}').replace(r'\$', r'$')


if not os.path.exists(path_relbert_pred):
    batch_size = 512

    def cap(_list):
        return [i.capitalize() for i in _list]


    def cos_similarity(a_, b_):
        inner = sum(list(map(lambda y: y[0] * y[1], zip(a_, b_))))
        norm_a = sum(list(map(lambda y: y * y, a_))) ** 0.5
        norm_b = sum(list(map(lambda y: y * y, b_))) ** 0.5
        return inner / (norm_b * norm_a)


    def get_prediction(_data, embedding_dict, model_name, capitalize=False):
        for single_data in _data:
            if capitalize:
                v_stem = embedding_dict[str(tuple(cap(single_data['stem'])))]
                v_choice = [embedding_dict[str(tuple(cap(c)))] for c in single_data['choice']]
            else:
                v_stem = embedding_dict[str(tuple(single_data['stem']))]
                v_choice = [embedding_dict[str(tuple(c))] for c in single_data['choice']]
            sims = [cos_similarity(v_stem, v) for v in v_choice]
            single_data['pred/{}'.format(model_name)] = sims.index(max(sims))
        return _data

    analogy_data = relbert.data.get_analogy_data()
    predictions = {}
    for data in ['bats_cap', 'google', 'bats']:
        if data == 'bats_cap':
            _, test = analogy_data['bats']
            all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in test]))
            all_pairs = [cap(i) for i in all_pairs]
        else:
            _, test = analogy_data[data]
            all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in test]))
        all_pairs = [tuple(i) for i in all_pairs]
        for n, i in zip(['manual', 'autoprompt', 'ptuning'],
                        models):
            model = relbert.RelBERT(i)
            embeddings = model.get_embedding(all_pairs, batch_size=batch_size)
            assert len(embeddings) == len(all_pairs)
            embeddings = {str(k_): v for k_, v in zip(all_pairs, embeddings)}
            test = get_prediction(test, embeddings, n, data == 'bats_cap')
        predictions[data] = test
    with open(path_relbert_pred, 'w') as f:
        json.dump(predictions, f)

with open(path_relbert_pred, 'r') as f:
    model_predictions = json.load(f)

high_level = []
low_level = []
meta_bats = {
            'Morphological': [
                'adj:comparative', 'adj:superlative', 'adj+ly', 'adj+ness', 'verb 3pSg:v+ed', 'verb v+ing:3pSg',
                'verb v+ing:v+ed', 'verb inf:3pSg', 'verb inf:v+ed', 'verb inf:v+ing', 'verb+able', 'verb+er',
                'verb+ment', 'verb+tion', 'un+adj', 'noun+less', 'over+adj', 'noun:plural', 're+verb'
            ],
            'Lexical': [
                'hypernyms:animals', 'hypernyms:misc', 'hyponyms:misc', 'antonyms:binary', 'antonyms:gradable',
                'meronyms:member', 'meronyms:part', 'meronyms:substance', 'synonyms:exact', 'synonyms:intensity',

            ],
            'Encyclopedic': [
                'UK city:county', 'animal:shelter', 'animal:sound', 'animal:young', 'country:capital',
                'country:language', 'male:female', 'name:nationality', 'name:occupation',
                'things:color',
            ]
        }
for i in ['google', 'bats']:
    val, test = we_predictions[i]
    df = pd.DataFrame(model_predictions[i]).sort_values(by=['stem'])
    df_we = pd.DataFrame(test).sort_values(by=['stem'])
    df['pred/fasttext'] = df_we['pred/fasttext']
    for method in ['manual', 'autoprompt', 'ptuning', 'fasttext']:
        df['accuracy/{}'.format(method)] = df['pred/{}'.format(method)] == df['answer']
    if i in ['bats', 'bats_cap']:
        df['prefix'] = df['prefix'].apply(lambda x: x.split(' [')[-1].split(']')[0].replace('_', ' ').
                                          replace(' - ', ':').replace(' reg', '').replace('V', 'v+').replace(' irreg', ''))
        df['prefix_high'] = df['prefix'].apply(lambda x: [k for k, v in meta_bats.items() if x in v][0])
    elif i == 'google':
        df['prefix_high'] = df['prefix'].apply(lambda x: 'Morphological' if 'gram' in x else "Semantic")
    # if i == 'bats_cap':
    #     g = df.groupby('prefix')
    #     df_new = g.aggregate('mean')[[c for c in df.columns if 'accuracy' in c]] * 100
    #     # case = ['UK city:county', 'country:capital', 'country:language', 'name:nationality', 'name:occupation']
    #     # bats_cap = df_new.T[case].T[['accuracy/fasttext', 'accuracy/manual',  'accuracy/autoprompt',  'accuracy/ptuning']]
    #     bats_cap = df_new[['accuracy/fasttext', 'accuracy/manual', 'accuracy/autoprompt', 'accuracy/ptuning']]
    #     bats_cap.index = [i + ' (cased)' for i in bats_cap.index]
    #     bats_cap.columns = ['FastText', 'Manual', 'AutoPrompt', 'P-tuning']
    # else:
    # g = df.groupby('prefix_high')
    g = df.groupby('prefix_high')
    df_new = g.aggregate('mean')[[c for c in df.columns if 'accuracy' in c]] * 100
    df_new = df_new[['accuracy/fasttext', 'accuracy/manual',  'accuracy/autoprompt',  'accuracy/ptuning']]
    df_new.columns = ['FastText', 'Manual', 'AutoPrompt', 'P-tuning']
    high_level.append(df_new.sort_index().round(1))

    g = df.groupby('prefix')
    df_new = g.aggregate('mean')[[c for c in df.columns if 'accuracy' in c]] * 100
    df_new = df_new[['accuracy/fasttext', 'accuracy/manual', 'accuracy/autoprompt', 'accuracy/ptuning']]
    df_new.columns = ['FastText', 'Manual', 'AutoPrompt', 'P-tuning']
    low_level.append(df_new.sort_index().round(1))

df = pd.concat(high_level)
df['Relation'] = df.index
df.index = ['Google'] * len(high_level[0]) + ['BATS'] * len(high_level[1])
df = df[['Relation', 'FastText', 'Manual', 'AutoPrompt', 'P-tuning']]
df.columns = [r'\textbf{' + i + r'}' for i in df.columns]
table = clean_latex(df.to_latex())
print('\n******* high level relation breakdown *******\n')
print(table)
print()

df = low_level[1].copy()
df.index.name = ''
df['Relation'] = [[k for k, v in meta_bats.items() if i in v][0] for i in df.index]
df['Relation_low'] = df.index
# df.index = df['Relation_high']
df = df.sort_values(by=['Relation', 'Relation_low'])
df.index = df['Relation']
df = df[['Relation_low', 'FastText', 'Manual', 'AutoPrompt', 'P-tuning']]
df.columns = [r'\textbf{' + i.replace('_low', '') + r'}' for i in df.columns]
table = clean_latex(df.to_latex())
print('\n******* low level relation breakdown of bats *******\n')
print(table)
print()
