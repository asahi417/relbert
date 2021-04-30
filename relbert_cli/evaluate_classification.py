import os
import logging
import argparse
from glob import glob
import pandas as pd
from relbert.evaluator.classification import evaluate
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def config(parser):
    parser.add_argument('-b', '--batch', help='batch size', default=512, type=int)
    parser.add_argument('--export-file', help='export file', required=True, type=str)
    return parser


def main():
    argument_parser = argparse.ArgumentParser(description='Evaluate on relation classification.')
    argument_parser = config(argument_parser)
    opt = argument_parser.parse_args()

    done_list = []
    full_result = []
    if os.path.exists(opt.export_file):
        df = pd.read_csv(opt.export_file, index_col=0)
        done_list = list(set(df['model'].values))
        full_result = [i.to_dict() for _, i in df.iterrows()]

    logging.info("RUN RELBERT")
    ckpts = sorted(glob('relbert_output/ckpt/*/epoch*'))
    for m in ckpts:
        if m in done_list:
            continue
        for if_both_direction in [True, False]:
            full_result += evaluate(relbert_ckpt=m, batch_size=opt.batch, both_direction=if_both_direction)
        pd.DataFrame(full_result).to_csv(opt.export_file)


if __name__ == '__main__':
    main()
