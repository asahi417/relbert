import pandas as pd
from datasets import load_dataset


data_names = ['google', 'bats', 'scan']
table_df = []
for i in data_names:
    data = load_dataset("relbert/analogy_questions", i, split='test')
    df_main = data.to_pandas()
    df_main['accuracy'] = pd.read_json(f"data/{i}.json")['full'].values.tolist()
    if i == 'google':
        df_main['prefix'] = df_main['prefix'].apply(lambda x: 'Morphological' if 'gram' in x else "Semantic")
    elif i == 'bats':
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
        df_main['prefix'] = df_main['prefix'].apply(lambda x: x.split(' [')[-1].split(']')[0].replace('_', ' ').replace(' - ', ':').replace(' reg', '').replace('V', 'v+').replace(' irreg', ''))
        df_main['prefix'] = df_main['prefix'].apply(lambda x: [k for k, v in meta_bats.items() if x in v][0])
    g = (df_main.groupby("prefix")["accuracy"].mean() * 100).round(1).T
    print(i, g)
