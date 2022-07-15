import os
import logging
import argparse
from glob import glob
import pandas as pd
from relbert.evaluator import evaluate_classification, evaluate_analogy

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='RelBERT evaluation on analogy/relation classification')
    parser.add_argument('-c', '--ckpt-dir', help='epoch of checkpoint', default='output/ckpt/*/*', type=str)
    parser.add_argument('--max-length', help='for vanilla LM', default=64, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=512, type=int)
    parser.add_argument('--export-file', help='export file', required=True, type=str)
    parser.add_argument('--type', help='test type (analogy/classification)', default='analogy', type=str)
    parser.add_argument('--distance-function', help='export file', default="cosine_similarity", type=str)
    parser.add_argument('--return-validation-loss', help='return validation loss', action='store_true')
    opt = parser.parse_args()

    done_list = []
    full_result = []
    os.makedirs(os.path.dirname(opt.export_file), exist_ok=True)
    if os.path.exists(opt.export_file):
        df = pd.read_csv(opt.export_file, index_col=0)
        done_list = list(set(df['model'].values))
        full_result = [i.to_dict() for _, i in df.iterrows()]

    ckpts = [i for i in sorted(glob(opt.ckpt_dir)) if os.path.isdir(i) and i not in done_list]
    logging.info(f"RUN {len(ckpts)} configurations")
    for m in ckpts:
        if opt.type == 'classification':
            full_result += evaluate_classification(relbert_ckpt=m, batch_size=opt.batch)
        elif opt.type == 'analogy':
            full_result += evaluate_analogy(relbert_ckpt=m, batch_size=opt.batch, max_length=opt.max_length,
                                            return_validation_loss=opt.return_validation_loss)
        else:
            raise ValueError('unknown test type: {}'.format(opt.type))
        pd.DataFrame(full_result).to_csv(opt.export_file)
