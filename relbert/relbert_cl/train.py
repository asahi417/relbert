""" RelBERT fine-tuning with NCE loss """
import argparse
import logging
from os.path import join as pj

import relbert


TEMP = "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Train RelBERT.')
    parser.add_argument('-m', '--model', help='', default='roberta-base', type=str)
    parser.add_argument('--max-length', help='', default=64, type=int)
    parser.add_argument('--mode', help='', default='average_no_mask', type=str)
    parser.add_argument('--data', help='', default='relbert/semeval2012_relational_similarity', type=str)
    parser.add_argument('--template-mode', help='', default='manual', type=str)
    parser.add_argument('-t', '--template', help='', default=TEMP, type=str)
    parser.add_argument('-l', '--loss-function', help='', default='nce_rank', type=str)
    parser.add_argument('--temperature-nce-type', help='', default='linear', type=str)
    parser.add_argument('--temperature-nce-constant', help='', default=1.0, type=float)
    parser.add_argument('--temperature-nce-min', help='', default=0.1, type=float)
    parser.add_argument('--temperature-nce-max', help='', default=10.0, type=float)
    parser.add_argument('-e', '--epoch', help='', default=1, type=int)
    parser.add_argument('-b', '--batch', help='', default=64, type=int)
    parser.add_argument('--n-sample', help='', default=640, type=int)
    parser.add_argument('--lr', help='', default=0.00002, type=float)
    parser.add_argument('--lr-decay', help='', action='store_true')
    parser.add_argument("--lr-warmup", help="", default=1, type=int)
    parser.add_argument('--random-seed', help='random seed', default=0, type=int)
    parser.add_argument("--weight-decay", help="", default=0, type=float)
    parser.add_argument('--exclude-relation', help="", nargs='+', default=None, type=str)
    parser.add_argument('--epoch-save', help='', default=1, type=int)
    parser.add_argument('-g', '--grad', help='', default=4, type=int)
    parser.add_argument('--export', help='', required=True, type=str)
    opt = parser.parse_args()

    trainer = relbert.Trainer(
        export=opt.export,
        model=opt.model,
        max_length=opt.max_length,
        mode=opt.mode,
        data=opt.data,
        template_mode=opt.template_mode,
        template=opt.template,
        loss_function=opt.loss_function,
        temperature_nce_type=opt.temperature_nce_type,
        temperature_nce_constant=opt.temperature_nce_constant,
        temperature_nce_min=opt.temperature_nce_min,
        temperature_nce_max=opt.temperature_nce_max,
        epoch=opt.epoch,
        batch=opt.batch,
        n_sample=opt.n_sample,
        lr=opt.lr,
        lr_decay=opt.lr_decay,
        lr_warmup=opt.lr_warmup,
        weight_decay=opt.weight_decay,
        random_seed=opt.random_seed,
        gradient_accumulation=opt.grad,
        exclude_relation=opt.exclude_relation)

    # add file handler
    logger = logging.getLogger()
    file_handler = logging.FileHandler(pj(trainer.export_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    logger.addHandler(file_handler)

    trainer.train(opt.epoch_save)

