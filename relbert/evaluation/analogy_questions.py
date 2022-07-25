import logging
from itertools import chain
import torch

from datasets import load_dataset

from ..lm import RelBERT


def cosine_similarity(a, b, zero_vector_mask: float = -100):
    norm_a = sum(map(lambda x: x * x, a)) ** 0.5
    norm_b = sum(map(lambda x: x * x, b)) ** 0.5
    if norm_b * norm_a == 0:
        return zero_vector_mask
    return sum(map(lambda x: x[0] * x[1], zip(a, b)))/(norm_a * norm_b)


def euclidean_distance(a, b):
    return sum(map(lambda x: (x[0] - x[1])**2, zip(a, b))) ** 0.5


def evaluate_analogy(relbert_ckpt: str = None,
                     max_length: int = 64,
                     batch_size: int = 64,
                     distance_function: str = 'cosine_similarity'):
    model = RelBERT(relbert_ckpt, max_length=max_length)
    assert model.is_trained, 'model is not trained'
    model.eval()
    result = {"distance_function": distance_function}
    with torch.no_grad():

        # Analogy test
        analogy_data = ['sat_full', 'sat', 'u2', 'u4', 'google', 'bats']
        for d in analogy_data:
            test = load_dataset('relbert/analogy_questions', d, split='test')
            all_pairs = list(chain(*list(chain(*[[test['stem']] + test['choice']]))))
            if d != 'sat_full':
                val = load_dataset('relbert/analogy_questions', d, split='validation')
                all_pairs += list(chain(*list(chain(*[[val['stem']] + val['choice']]))))
            else:
                val = None
            logging.info(f'\t * data: {d}')
            # preprocess data
            all_pairs = [tuple(i) for i in all_pairs]
            embeddings = model.get_embedding(all_pairs, batch_size=batch_size)
            assert len(embeddings) == len(all_pairs), f"{len(embeddings)} != {len(all_pairs)}"
            embeddings_dict = {str(tuple(k_)): v for k_, v in zip(all_pairs, embeddings)}

            def prediction(_data):
                accuracy = []
                for single_data in _data:
                    v_stem = embeddings_dict[str(tuple(single_data['stem']))]
                    v_choice = [embeddings_dict[str(tuple(c))] for c in single_data['choice']]
                    if distance_function == "cosine_similarity":
                        sims = [cosine_similarity(v_stem, v) for v in v_choice]
                    elif distance_function == "euclidean_distance":
                        sims = [euclidean_distance(v_stem, v) for v in v_choice]
                    else:
                        raise ValueError(f'unknown distance function {distance_function}')
                    pred = sims.index(max(sims))
                    if sims[pred] == -100:
                        raise ValueError('failed to compute similarity')
                    accuracy.append(single_data['answer'] == pred)
                return sum(accuracy) / len(accuracy)

            # get prediction
            result[f'{d}/test'] = prediction(test)
            if val is not None:
                result[f'{d}/valid'] = prediction(val)
    result['sat_full'] = result.pop('sat_full/test')
    logging.info(str(result))
    del model
    return result



