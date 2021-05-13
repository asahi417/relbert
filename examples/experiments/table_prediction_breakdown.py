import json
import os
from itertools import chain
import pandas as pd
import relbert


path_relbert_pred = 'asset/prediction.relbert.json'
path_fasttext_pred = {'bats': 'asset/prediction.bats.fasttext.csv', 'google': 'asset/prediction.google.fasttext.csv'}

def cap(_list):
    return [i.capitalize() for i in _list]


if not os.path.exists(path_relbert_pred):
    batch_size = 512

    def cos_similarity(a_, b_):
        inner = sum(list(map(lambda y: y[0] * y[1], zip(a_, b_))))
        norm_a = sum(list(map(lambda y: y * y, a_))) ** 0.5
        norm_b = sum(list(map(lambda y: y * y, b_))) ** 0.5
        return inner / (norm_b * norm_a)


    def get_prediction(_data, embedding_dict, model_name):
        for single_data in _data:
            v_stem = embedding_dict[str(tuple(single_data['stem']))]
            v_choice = [embedding_dict[str(tuple(c))] for c in single_data['choice']]
            sims = [cos_similarity(v_stem, v) for v in v_choice]
            single_data['pred/{}'.format(model_name)] = sims.index(max(sims))
        return _data


    analogy_data = relbert.data.get_analogy_data()
    predictions = {}
    for data in ['google', 'bats', 'bats_cap']:
        if data == 'bats_cap':
            _, test = analogy_data['bats']
            all_pairs = list(chain(*[[cap(o['stem'])] + cap(o['choice']) for o in test]))
        else:
            _, test = analogy_data[data]
            all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in test]))
        all_pairs = [tuple(i) for i in all_pairs]
        for n, i in zip(['manual', 'autoprompt', 'ptuning'],
                        ["asahi417/relbert_roberta_custom_c", "asahi417/relbert_roberta_autoprompt", "asahi417/relbert_roberta_ptuning"]):
            model = relbert.RelBERT(i)
            embeddings = model.get_embedding(all_pairs, batch_size=batch_size)
            assert len(embeddings) == len(all_pairs)
            embeddings = {str(k_): v for k_, v in zip(all_pairs, embeddings)}
            test = get_prediction(test, embeddings, n)
        predictions[data] = test
    with open(path_relbert_pred, 'w') as f:
        json.dump(predictions, f)
else:
    with open(path_relbert_pred, 'r') as f:
        predictions = json.load(f)

for i in ['google', 'bats']:
    df_fasttext = pd.read_csv(path_fasttext_pred[i], index_col=0).sort_values(by=['stem'])
    df = pd.DataFrame(predictions[i]).sort_values(by=['stem'])
    df['pred/fasttext'] = df_fasttext.prediction
    for method in ['manual', 'autoprompt', 'ptuning', 'fasttext']:
        df['accuracy/{}'.format(method)] = df['pred/{}'.format(method)] == df['answer']
    if i == 'bats':
        df['prefix'] = df['prefix'].apply(lambda x: x.split(' [')[-1].split(']')[0].replace('_', ' ').
                                          replace(' - ', ':').replace(' reg', '').replace('V', 'v+').replace(' irreg', ''))
        meta = {
            'Morphological': [
                'adj:comparative', 'adj:superlative', 'adj+ly', 'adj+ness', 'verb 3pSg:v+ed', 'verb v+ing:3pSg',
                'verb v+ing:v+ed', 'verb inf:3pSg', 'verb inf:v+ed', 'verb inf:v+ing', 'verb+able', 'verb+er',
                'verb+ment', 'verb+tion', 'un+adj', 'noun+less', 'over+adj',
            ],
            'Lexical': [
                'hypernyms:animals', 'hypernyms:misc', 'hyponyms:misc', 'antonyms:binary', 'antonyms:gradable',
                'meronyms:member', 'meronyms:part', 'meronyms:substance', 'synonyms:exact', 'synonyms:intensity',
                're+verb'
            ],
            'Encyclopedic': [
                'UK city:county', 'animal:shelter', 'animal:sound', 'animal:young', 'country:capital',
                'country:language', 'male:female', 'name:nationality', 'name:occupation', 'noun:plural',
                'things:color',
            ]
        }
        df['prefix_high'] = df['prefix'].apply(lambda x: [k for k, v in meta.items() if x in v][0])
    elif i == 'google':
        df['prefix_high'] = df['prefix'].apply(lambda x: 'Morphological' if 'gram' in x else "Semantic")
    g = df.groupby('prefix')
    df_new = g.aggregate('mean')[[c for c in df.columns if 'accuracy' in c]] * 100
    df_new['relation_class'] = [df[df.prefix == c].prefix_high.values[0] for c in df_new.index]
    df_new.to_csv('relbert_output/eval/summary/prediction.{}.csv'.format(i))
