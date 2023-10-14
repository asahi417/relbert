import os
from random import seed
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from relbert import RelBERT

os.makedirs("results", exist_ok=True)
plt.rcParams.update({'font.size': 16})  # must set in top
styles = ['o-', 'o--', 'o:', 's-', 's--', 's:', '^-', '^--', '^:', "X-", "X--", "X:"]
seed(1)
colors = list(mpl.colormaps['Dark2'].colors)


def cosine_similarity(a, b):
    norm_a = sum(map(lambda x: x * x, a)) ** 0.5
    norm_b = sum(map(lambda x: x * x, b)) ** 0.5
    return sum(map(lambda x: x[0] * x[1], zip(a, b)))/(norm_a * norm_b)


data_names = ['u2', 'u4', 'google', 'bats', 'scan']
table_df = []
for i in data_names:
    df_main = None
    for size in ['base', 'large']:
        model = RelBERT(f"relbert/relbert-roberta-{size}")
        if not os.path.exists(f"results/{i}.{size}.csv"):

            data = load_dataset("relbert/analogy_questions", i, split='test')
            v_stem = model.get_embedding(data['stem'], batch_size=1024)

            choice_flat = []
            choice_index = []
            for n, c in enumerate(data['choice']):
                choice_flat += c
                choice_index += [n] * len(c)
            v_choice_flat = model.get_embedding(choice_flat, batch_size=1024)

            v_choice = [[v for v, c_i in zip(v_choice_flat, choice_index) if c_i == _i] for _i in range(len(data['choice']))]
            assert len(v_choice) == len(v_stem)
            sims = [[cosine_similarity(_c, s) for _c in c] for c, s in zip(v_choice, v_stem)]
            pred = [s.index(max(s)) for s in sims]
            acc = [int(a == p) for a, p in zip(data['answer'], pred)]
            df = pd.DataFrame([data['prefix'], acc], index=['prefix', 'accuracy']).T
            df.to_csv(f"results/{i}.{size}.csv")
        df = pd.read_csv(f"results/{i}.{size}.csv", index_col=0)
        if df_main is None:
            df_main = df
            df_main[size] = df_main.pop("accuracy")
        else:
            df_main[size] = df.pop("accuracy")
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
        df_main['prefix'] = df_main['prefix'].apply(lambda x: x.split(' [')[-1].split(']')[0].replace('_', ' ').
                                          replace(' - ', ':').replace(' reg', '').replace('V', 'v+').replace(
            ' irreg', ''))
        df_main['prefix'] = df_main['prefix'].apply(lambda x: [k for k, v in meta_bats.items() if x in v][0])
    g = (df_main.groupby("prefix").mean() * 100).round(1).T
    g.columns.name = ''
    g.index = ["RelBERT (base)", "RelBERT (large)"]
    if i in ['u2', 'u4']:
        if i == 'u2':
            g = g[['grade4',  'grade5',  'grade6',  'grade7',  'grade8',  'grade9', 'grade10',  'grade11',  'grade12']]
        else:
            g = g[['high-beginning', 'low-intermediate', 'high-intermediate', 'low-advanced', 'high-advanced']]
        g.T.plot(grid=True, style=styles, color=colors, figsize=(10, 6))
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"results/{size}.{i}.png", bbox_inches="tight", dpi=600)
    else:
        table_df.append(g)

print(pd.concat(table_df, axis=1).to_latex())