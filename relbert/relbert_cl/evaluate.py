import os
import json
import logging
import argparse
from relbert import evaluate_analogy, evaluate_classification, evaluate_relation_mapping

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main_analogy():
    parser = argparse.ArgumentParser(description='RelBERT evaluation on analogy question')
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('-o', '--output-file', help='export file', required=True, type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=512, type=int)
    parser.add_argument('-l', '--max-length', help='for vanilla LM', default=64, type=int)
    parser.add_argument('-d', '--data', help='target analogy', nargs='+', default=None, type=str)
    parser.add_argument('-s', '--split', help='target analogy', default='test', type=str)
    parser.add_argument('--aggregation-mode', help='aggregation mode (for vanilla LM)', default='average_no_mask', type=str)
    parser.add_argument('-t', '--template', help='template (for vanilla LM)', default=None, type=str)
    parser.add_argument('--overwrite', help='', action='store_true')
    parser.add_argument('--reverse-pair', help='', action='store_true')
    parser.add_argument('--bi-direction-pair', help='', action='store_true')
    opt = parser.parse_args()

    data = ['sat_full', 'sat', 'u2', 'u4', 'google', 'bats'] if opt.data is None else opt.data

    if os.path.exists(opt.output_file):
        with open(opt.output_file) as f:
            tmp_out = json.load(f)
        if not opt.overwrite:
            done_list = [k for k in tmp_out.keys() if k.endswith(opt.split)]
            data = [i for i in data if i not in done_list]
    else:
        tmp_out = {}

    out = evaluate_analogy(
        target_analogy=data,
        relbert_ckpt=opt.model,
        max_length=opt.max_length,
        batch_size=opt.batch,
        target_analogy_split=opt.split,
        distance_function='cosine_similarity',
        reverse_pair=opt.reverse_pair,
        bi_direction_pair=opt.bi_direction_pair,
        aggregation_mode=opt.aggregation_mode,
        template=opt.template
    )
    tmp_out.update(out)
    if os.path.dirname(opt.output_file) != '':
        os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)
    with open(opt.output_file, 'w') as f:
        json.dump(tmp_out, f)


def main_classification():
    parser = argparse.ArgumentParser(description='RelBERT evaluation on lexical relation classification')
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('-o', '--output-file', help='export file', required=True, type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=512, type=int)
    parser.add_argument('-l', '--max-length', help='for vanilla LM', default=64, type=int)
    parser.add_argument('--target-relation', help='target relation', default=None, type=str)
    parser.add_argument('--overwrite', help='', action='store_true')
    opt = parser.parse_args()

    if not opt.overwrite and os.path.exists(opt.output_file):
        logging.info(f"{opt.output_file} exists, skip")
        return

    out = evaluate_classification(
        relbert_ckpt=opt.model,
        max_length=opt.max_length,
        batch_size=opt.batch,
        target_relation=opt.target_relation,
    )
    if os.path.dirname(opt.output_file) != '':
        os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)
    with open(opt.output_file, 'w') as f:
        json.dump(out, f)


def main_relation_mapping():
    parser = argparse.ArgumentParser(description='RelBERT evaluation on relation mapping')
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('-o', '--output-file', help='export file', required=True, type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=512, type=int)
    parser.add_argument('--data', default="relbert/relation_mapping", type=str)
    parser.add_argument('--overwrite', help='', action='store_true')
    opt = parser.parse_args()

    if not opt.overwrite and os.path.exists(opt.output_file):
        logging.info(f"{opt.output_file} exists, skip")
        return

    mean_accuracy, sims_full, _ = evaluate_relation_mapping(
        relbert_ckpt=opt.model,
        batch_size=opt.batch,
        dataset=opt.data
    )
    if os.path.dirname(opt.output_file) != '':
        os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)
    with open(opt.output_file, 'w') as f:
        json.dump({"accuracy": mean_accuracy, "sims_full": sims_full}, f)


# def main_analogy_relation_data():
#     parser = argparse.ArgumentParser(description='RelBERT evaluation on training data converted into analogy question')
#     parser.add_argument('-m', '--model', help='model', required=True, type=str)
#     parser.add_argument('-o', '--output-file', help='export file', required=True, type=str)
#     parser.add_argument('-b', '--batch', help='batch size', default=512, type=int)
#     parser.add_argument('-l', '--max-length', help='for vanilla LM', default=64, type=int)
#     parser.add_argument('--data', help='target data', default='relbert/semeval2012_relational_similarity', type=str)
#     parser.add_argument('--split', help='target split', default='validation', type=str)
#     parser.add_argument('--aggregation-mode', help='aggregation mode (for vanilla LM)', default='average_no_mask', type=str)
#     parser.add_argument('-t', '--template', help='template (for vanilla LM)', default=None, type=str)
#     parser.add_argument('--overwrite', help='', action='store_true')
#     parser.add_argument('--reverse-pair', help='', action='store_true')
#     parser.add_argument('--bi-direction-pair', help='', action='store_true')
#     opt = parser.parse_args()
#
#     if not opt.overwrite and os.path.exists(opt.output_file):
#         logging.info(f"{opt.output_file} exists, skip")
#         return
#
#     tmp_data = load_dataset(opt.data, split=opt.split)
#     analogy_data = [{"stem": i['positives'][0], "choice": i["negatives"] + [i['positives'][1]], "answer": 2,
#                      "prefix": i["relation_type"]} for i in tmp_data]
#     out = evaluate_analogy(
#         relbert_ckpt=opt.model,
#         max_length=opt.max_length,
#         batch_size=opt.batch,
#         distance_function='cosine_similarity',
#         reverse_pair=opt.reverse_pair,
#         bi_direction_pair=opt.bi_direction_pair,
#         aggregation_mode=opt.aggregation_mode,
#         template=opt.template,
#         hf_dataset=Dataset.from_list(analogy_data),
#         hf_dataset_name=opt.data,
#         hf_dataset_split=opt.split
#     )
#     if os.path.dirname(opt.output_file) != '':
#         os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)
#     with open(opt.output_file, 'w') as f:
#         json.dump(out, f)
