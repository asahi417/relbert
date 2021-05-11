import json
import os
from itertools import chain
import pandas as pd
import relbert
import seaborn as sns
from matplotlib import pylab as plt

plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
sns.set_theme(style="darkgrid")


if not os.path.exists('relbert_output/ablation_study/prediction_breakdown/prediction.json'):
    os.makedirs('relbert_output/ablation_study/prediction_breakdown', exist_ok=True)
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
    for data in ['google', 'bats']:
        _, test = analogy_data[data]
        for n, i in zip(['manual', 'autoprompt', 'ptuning'], ["asahi417/relbert_roberta_custom_c", "asahi417/relbert_roberta_autoprompt", "asahi417/relbert_roberta_ptuning"]):
            model = relbert.RelBERT(i)
            predictions = {}
            # preprocess data
            all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in test]))
            all_pairs = [tuple(i) for i in all_pairs]
            embeddings = model.get_embedding(all_pairs, batch_size=batch_size)
            assert len(embeddings) == len(all_pairs)
            embeddings = {str(k_): v for k_, v in zip(all_pairs, embeddings)}
            predictions[data] = get_prediction(test, embeddings, n)

    with open('relbert_output/ablation_study/prediction_breakdown/prediction.json', 'w') as f:
        json.dump(predictions, f)

else:
    with open('relbert_output/ablation_study/prediction_breakdown/prediction.json', 'r') as f:
        predictions = json.load(f)

for i in ['google', 'bats']:
    df = pd.DataFrame(predictions[i])

# fig = plt.figure()
# fig.clear()
# df['data'] = [i.replace('bats', 'BATS').replace('u2', 'U2').replace('u4', 'U4').replace('google', 'Google').replace('sat', 'SAT') for i in df.data]
# df['accuracy'] = df['custom'].astype(float)
# ax = sns.barplot(data=df, x='data', y='accuracy', hue='lm', order=['SAT', 'U2', 'U4', 'BATS', 'Google'], hue_order=['ALBERT', 'BERT', 'RoBERTa'])
# # ax.set(ylim=(0, 100))
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels)
# plt.setp(ax.get_legend().get_texts(), fontsize='15')
# ax.set_xlabel(None)
# ax.set_ylabel('Accuracy', fontsize=15)
# ax.tick_params(labelsize=15)
# fig = ax.get_figure()
# plt.tight_layout()
# fig.savefig('./relbert_output/eval/summary/fig.lm.comparison.png')
# plt.close()
