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
                     distance_function: str = 'cosine_similarity',
                     reverse_pair: bool = False,
                     bi_direction_pair: bool = False,
                     target_analogy: str = None):
    model = RelBERT(relbert_ckpt, max_length=max_length)
    assert model.is_trained, 'model is not trained'
    target_analogy = ['sat_full', 'sat', 'u2', 'u4', 'google', 'bats'] if target_analogy is None else [target_analogy]
    model.eval()
    result = {"distance_function": distance_function}
    with torch.no_grad():

        # Analogy test
        for d in target_analogy:
            test = load_dataset('relbert/analogy_questions', d, split='test')
            all_pairs = list(chain(*list(chain(*[[test['stem']] + test['choice']]))))
            val = None
            if d != 'sat_full':
                val = load_dataset('relbert/analogy_questions', d, split='validation')
                all_pairs += list(chain(*list(chain(*[[val['stem']] + val['choice']]))))
            if reverse_pair:
                all_pairs = [[b, a] for a, b in all_pairs]
            elif bi_direction_pair:
                all_pairs += [[b, a] for a, b in all_pairs]
            logging.info(f'\t * data: {d}')

            # preprocess data
            all_pairs = [tuple(i) for i in all_pairs]
            embeddings = model.get_embedding(all_pairs, batch_size=batch_size)
            assert len(embeddings) == len(all_pairs), f"{len(embeddings)} != {len(all_pairs)}"
            embeddings_dict = {str(tuple(k_)): v for k_, v in zip(all_pairs, embeddings)}

            def get_sim(_single_data, _reverse):
                v_s = embeddings_dict[str(tuple(_single_data['stem'] if not _reverse else _single_data['stem'][::-1]))]
                v_c = [embeddings_dict[str(tuple(c if not _reverse else c[::-1]))] for c in _single_data['choice']]
                if distance_function == "cosine_similarity":
                    return [cosine_similarity(v_s, v) for v in v_c]
                elif distance_function == "euclidean_distance":
                    return [euclidean_distance(v_s, v) for v in v_c]
                else:
                    raise ValueError(f'unknown distance function {distance_function}')

            def prediction(_data):
                accuracy = []
                for single_data in _data:
                    if bi_direction_pair:
                        sim = get_sim(single_data, False)
                        sim_r = get_sim(single_data, True)
                        sims = [s * s_r for s, s_r in zip(sim, sim_r)]
                    else:
                        sims = get_sim(single_data, reverse_pair)
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



