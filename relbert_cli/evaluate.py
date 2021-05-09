import os
import logging
import argparse
from glob import glob
import pandas as pd
from relbert.evaluator import evaluate_classification, evaluate_analogy
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def config(parser):
    parser.add_argument('-b', '--batch', help='batch size', default=512, type=int)
    parser.add_argument('--export-file', help='export file', required=True, type=str)
    parser.add_argument('-c', '--ckpt-dir', help='epoch of checkpoint', default='relbert_output/ckpt/*/*', type=str)
    parser.add_argument('--type', help='test type (analogy/classification)', default='analogy', type=str)
    parser.add_argument('-l', '--max-length', help='for vanilla LM', default=64, type=int)
    parser.add_argument('-m', '--mode', help='for vanilla LM', default='average_no_mask', type=str)
    parser.add_argument('-t', '--template-type', help='for vanilla LM', default=None, type=str)
    parser.add_argument('--vanilla-lm', action='store_true')
    return parser


def main():
    argument_parser = argparse.ArgumentParser(description='RelBERT evaluation on analogy/relation classification')
    argument_parser = config(argument_parser)
    opt = argument_parser.parse_args()

    done_list = []
    full_result = []
    if os.path.exists(opt.export_file):
        df = pd.read_csv(opt.export_file, index_col=0)
        done_list = list(set(df['model'].values))
        full_result = [i.to_dict() for _, i in df.iterrows()]

    if opt.vanilla_lm:
        logging.info('RUN Vanilla LM')
        ckpts = opt.ckpt_dir.split(',')
    else:
        logging.info("RUN RelBERT")
        ckpts = [i for i in sorted(glob(opt.ckpt_dir)) if os.path.exists(i)]
    for m in ckpts:
        if m in done_list:
            continue
        if opt.type == 'classification':
            full_result += evaluate_classification(relbert_ckpt=m, batch_size=opt.batch)
        elif opt.type == 'analogy':
            full_result += evaluate_analogy(relbert_ckpt=m, batch_size=opt.batch,
                                            mode=opt.mode, template_type=opt.template_type, max_length=opt.max_length)
        else:
            raise ValueError('unknown test type: {}'.format(opt.type))
        pd.DataFrame(full_result).to_csv(opt.export_file)


if __name__ == '__main__':
    main()
