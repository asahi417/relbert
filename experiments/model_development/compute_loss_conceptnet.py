import json
import logging
import argparse
import os
from os.path import join as pj
from relbert.evaluation import evaluate_classification, evaluate_analogy, evaluate_validation_loss, \
    evaluate_relation_mapping

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='RelBERT evaluation on analogy/relation classification')
    parser.add_argument('-c', '--ckpt-dir', help='epoch of checkpoint', required=True, type=str)
    parser.add_argument('-d', '--data', help='data', required=True, type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=512, type=int)
    parser.add_argument('--overwrite', help='', action='store_true')
    opt = parser.parse_args()

    export_dir = opt.ckpt_dir if opt.export_dir is None else opt.export_dir
    output_file = pj(export_dir, f'{opt.type}.json')
    output_file = output_file.replace('.json', f'.{os.path.basename(opt.data)}.json')
    if os.path.exists(output_file):
        if opt.overwrite:
            logging.warning(f'overwrite the result {output_file}')
        else:
            logging.info(f'result already exists at {output_file}. add `--overwrite` to overwrite the result.')
            return
    result_ = evaluate_validation_loss(
        validation_data=opt.data,
        relbert_ckpt=opt.ckpt_dir,
        batch_size=opt.batch,
        max_length=opt.max_length,
        split=opt.split,
        exclude_relation=opt.exclude_relation
    )
    if os.path.exists(output_file):
        with open(output_file) as f:
            result = json.load(f)
        result.update(result_)
    else:
        result = result_
    with open(output_file, 'w') as f:
        json.dump(result, f)
