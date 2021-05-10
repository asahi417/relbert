import json
import os
from itertools import chain
import numpy as np
import relbert

os.makedirs('relbert_output/ablation_study/prediction_breakdown', exist_ok=True)
bi_direction = False
batch_size = 512

if not os.path.exists('relbert_output/ablation_study/prediction_breakdown/prediction.json'):
    def cos_similarity(a_, b_):
        inner = sum(list(map(lambda y: y[0] * y[1], zip(a_, b_))))
        norm_a = sum(list(map(lambda y: y * y, a_))) ** 0.5
        norm_b = sum(list(map(lambda y: y * y, b_))) ** 0.5
        return inner / (norm_b * norm_a)


    def get_prediction(_data, embedding_dict):
        for single_data in _data:
            v_stem = embedding_dict[str(tuple(single_data['stem']))]
            v_choice = [embedding_dict[str(tuple(c))] for c in single_data['choice']]
            if bi_direction:
                reverse_key = str((single_data['stem'][1], single_data['stem'][0]))
                v_stem = np.concatenate([v_stem, embedding_dict[reverse_key]])
                v_choice_r = [embedding_dict[str((c[1], c[0]))] for c in single_data['choice']]
                v_choice = [np.concatenate(i) for i in zip(v_choice, v_choice_r)]
            sims = [cos_similarity(v_stem, v) for v in v_choice]
            single_data['pred'] = sims.index(max(sims))
        return data

    model = relbert.RelBERT("asahi417/relbert_roberta_custom_c")
    analogy_data = relbert.data.get_analogy_data()
    predictions = {}
    for data in ['google', 'bats']:
        _, test = analogy_data[data]
        # preprocess data
        all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in test]))
        if bi_direction:
            all_pairs += [(i[1], i[0]) for i in all_pairs]
        embeddings = model.get_embedding(all_pairs, batch_size=batch_size)
        assert len(embeddings) == len(all_pairs)
        embeddings = {str(k_): v for k_, v in zip(all_pairs, embeddings)}
        predictions[data] = get_prediction(test, embeddings)


    with open('relbert_output/ablation_study/prediction_breakdown/prediction.json', 'w') as f:
        json.dump(predictions, f)
else:
    with open('relbert_output/ablation_study/prediction_breakdown/prediction.json', 'r') as f:
        predictions = json.load(f)