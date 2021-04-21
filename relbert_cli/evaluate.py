""" Evaluate LM on relational knowledge. """
import os
import json
import argparse
import logging
from glob import glob
import relbert


def config(parser):
    parser.add_argument('-c', '--ckpt', help='checkpoint dir', default=None)
    parser.add_argument('-b', '--batch', help='batch size', default=1024, type=int)
    parser.add_argument('--cache-dir', help='cache directory to store dataset', default=None, type=str)
    parser.add_argument('--num-workers', help='workers for dataloder', default=1, type=int)
    parser.add_argument('--debug', help='log level', action='store_true')
    parser.add_argument('--export-file', help='export file', required=True, type=str)
    parser.add_argument('-m', '--model', help='vanilla language model', default='roberta-large', type=str)
    parser.add_argument('-l', '--max-length', help='length', default=64, type=int)
    parser.add_argument('--mode', help='lm mode', default='average', type=str)
    parser.add_argument('--test-type', help='test data', default='analogy', type=str)
    parser.add_argument('-t', '--template-type', help='template type or path to generated prompt file',
                        default='a', type=str)

    # parser.add_argument('-n', '--in-batch-negative', help='in batch negative', action='store_true')
    # parser.add_argument('--mse-margin', help='contrastive loss margin', default=1, type=int)
    parser.add_argument('--data', help='dataset', default='semeval2012', type=str)

    return parser


def main():
    argument_parser = argparse.ArgumentParser(description='Evaluate LM on relational knowledge.')
    argument_parser = config(argument_parser)
    opt = argument_parser.parse_args()
    # logging
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('RelBERT Evaluation')
    if opt.ckpt is not None:
        ckpt = opt.ckpt.split(',')
        logging.info('* evaluate trained RelBERT: {}'.format(ckpt))
    else:
        ckpt = opt.model.split(',')
        logging.info('* evaluate vanilla LM: {}'.format(ckpt))

    for n, i in enumerate(ckpt):
        logging.info('## start evaluation {}/{}: {} ##'.format(n + 1, len(ckpt), i))
        if os.path.exists('{}/trainer_config.json'.format(i)):
            with open('{}/trainer_config.json'.format(i), 'r') as f:
                trainer_config = json.load(f)
            shared_config = {
                'data': trainer_config['data'],
                'softmax_loss': trainer_config['softmax_loss'],
                'in_batch_negative': trainer_config['in_batch_negative'],
                'parent_contrast': trainer_config['parent_contrast'],
                'mse_margin': trainer_config['mse_margin']
            }
            relbert.evaluate(
                model=[os.path.dirname(i) for i in glob('{}/*/pytorch_model.bin'.format(i))],
                max_length=trainer_config['max_length'],
                test_type=opt.test_type,
                export_file=opt.export_file,
                cache_dir=opt.cache_dir,
                batch=opt.batch,
                num_worker=opt.num_workers,
                validation_data=trainer_config['data'],
                # mse_margin=trainer_config['mse_margin'],
                # in_batch_negative=trainer_config['in_batch_negative'],
                shared_config=shared_config
            )
        else:
            shared_config = {
                'data': None, 'softmax_loss': None, 'in_batch_negative': None, 'parent_contrast': None,
                'mse_margin': None}
            relbert.evaluate(
                model=[i],
                max_length=opt.max_length,
                template_type=opt.template_type,
                mode=opt.mode,
                test_type=opt.test_type,
                export_file=opt.export_file,
                cache_dir=opt.cache_dir,
                batch=opt.batch,
                num_worker=opt.num_workers,
                validation_data=opt.data,
                # mse_margin=opt.mse_margin,
                # in_batch_negative=False,
                shared_config=shared_config
            )


if __name__ == '__main__':
    main()
