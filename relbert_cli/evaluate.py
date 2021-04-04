""" Evaluate LM on relational knowledge. """
import os
import json
import argparse
import logging
from glob import glob
import relbert


def config(parser):
    parser.add_argument('-c', '--ckpt', help='checkpoint dir', default='./ckpt/relbert')
    parser.add_argument('-b', '--batch', help='batch size', default=32, type=int)

    parser.add_argument('--cache-dir', help='cache directory to store dataset', default=None, type=str)
    parser.add_argument('--num-workers', help='workers for dataloder', default=1, type=int)
    parser.add_argument('--debug', help='log level', action='store_true')
    parser.add_argument('--export-file', help='export file', default='./eval/relbert.eval.csv', type=str)

    parser.add_argument('-m', '--model', help='vanilla language model', default=None, type=str)
    parser.add_argument('-l', '--max-length', help='length', default=64, type=int)
    parser.add_argument('--mode', help='lm mode', default='mask', type=str)

    parser.add_argument('--test-type', help='test data', default='analogy', type=str)
    parser.add_argument('-t', '--template-type', help='template type', default='a', type=str)
    return parser


def main():
    argument_parser = argparse.ArgumentParser(description='Evaluate LM on relational knowledge.')
    argument_parser = config(argument_parser)
    opt = argument_parser.parse_args()
    # logging
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('RelBERT Evaluation')
    if opt.model is not None:
        ckpt = opt.model.split(',')
        logging.info('* evaluate vanilla LM: {}'.format(ckpt))
    else:
        ckpt = glob('{}/*'.format(opt.ckpt))
        logging.info('* evaluate trained RelBERT: {}'.format(ckpt))

    for n, i in enumerate(ckpt):
        logging.info('## start evaluation {}/{}: {} ##'.format(n + 1, len(ckpt), i))
        if os.path.exists('{}/trainer_config.json'.format(i)):
            with open('{}/trainer_config.json'.format(i), 'r') as f:
                trainer_config = json.load(f)
            shared_config = {
                'softmax_loss': trainer_config['softmax_loss'],
                'in_batch_negative': trainer_config['in_batch_negative'],
                'parent_contrast': trainer_config['parent_contrast'],
                'mse_margin': trainer_config['mse_margin']
            }
            model = [os.path.dirname(i) for i in glob('{}/*/pytorch_model.bin'.format(i))]
            relbert.evaluate(
                model=model,
                max_length=trainer_config['max_length'],
                test_type=opt.test_type,
                export_file=opt.export_file,
                cache_dir=opt.cache_dir,
                batch=opt.batch,
                num_worker=opt.num_workers,
                shared_config=shared_config
            )
        else:
            shared_config = {
                'softmax_loss': None, 'in_batch_negative': None, 'parent_contrast': None, 'mse_margin': None}
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
                shared_config=shared_config
            )


if __name__ == '__main__':
    main()
