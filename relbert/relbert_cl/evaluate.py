import json
import logging
import argparse

from relbert.evaluation import evaluate_classification, evaluate_analogy, evaluate_validation_loss

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='RelBERT evaluation on analogy/relation classification')
    parser.add_argument('-c', '--ckpt-dir', help='epoch of checkpoint', required=True, type=str)
    parser.add_argument('--export-dir', help='export file', default=None, type=str)
    parser.add_argument('--max-length', help='for vanilla LM', default=64, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=512, type=int)
    parser.add_argument('--type', help='test type (analogy/classification/validation_loss)', default='analogy', type=str)
    parser.add_argument('--distance-function', help='export file', default="cosine_similarity", type=str)
    opt = parser.parse_args()

    export_dir = opt.ckpt_dir if opt.export_dir is None else opt.export_dir
    if opt.type == 'classification':
        result = evaluate_classification(relbert_ckpt=opt.ckpt_dir, batch_size=opt.batch)
    elif opt.type == 'analogy':
        result = evaluate_analogy(relbert_ckpt=opt.ckpt_dir, batch_size=opt.batch, max_length=opt.max_length)
    elif opt.type == 'validation_loss':
        result = evaluate_validation_loss(relbert_ckpt=opt.ckpt_dir, batch_size=opt.batch, max_length=opt.max_length)
    else:
        raise ValueError(f'unknown test type: {opt.type}')
    with open(f'{export_dir}/{opt.type}.json', 'w') as f:
        json.dump(result, f)