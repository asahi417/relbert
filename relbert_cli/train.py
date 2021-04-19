""" Train RelBERT model. """
import argparse
import logging
import relbert


def config(parser):
    # optimization
    parser.add_argument('-s', '--softmax-loss', help='softmax loss', action='store_true')
    parser.add_argument('-n', '--in-batch-negative', help='in batch negative', action='store_true')
    parser.add_argument('-p', '--parent-contrast', help='hierarchical contrastive loss', action='store_true')
    parser.add_argument('-e', '--epoch', help='training epochs', default=5, type=int)
    parser.add_argument('--mse-margin', help='contrastive loss margin', default=1, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=16, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.00005, type=float)
    parser.add_argument('--random-seed', help='random seed', default=0, type=int)
    parser.add_argument('--lr-decay', help='linear decay of learning rate after warmup', action='store_true')
    parser.add_argument("--lr-warmup", help="linear warmup of lr", default=100, type=int)
    parser.add_argument("--weight-decay", help="l2 penalty for weight decay", default=0, type=float)
    parser.add_argument('--optimizer', help='optimizer `adam`/`adamax`/`adam`', default='adam', type=str)
    parser.add_argument("--momentum", help="sgd momentum", default=0.9, type=float)
    # training environment
    parser.add_argument('--cache-dir', help='cache directory to store dataset', default=None, type=str)
    parser.add_argument('--num-workers', help='workers for dataloder', default=1, type=int)
    parser.add_argument('--fp16', help='fp16 for training', action='store_true')
    parser.add_argument('--epoch-save', help='interval to save model weight', default=1, type=int)
    parser.add_argument('--debug', help='log level', action='store_true')
    parser.add_argument('--export', help='directory to export model weight file', required=True, type=str)
    # language model
    parser.add_argument('-m', '--model', help='language model', default='roberta-large', type=str)
    parser.add_argument('-l', '--max-length', help='length', default=64, type=int)
    parser.add_argument('--mode', help='lm mode', default='average', type=str)
    # data
    parser.add_argument('--data', help='dataset', default='semeval2012', type=str)
    parser.add_argument('--n-sample', help='sample size', default=5, type=int)
    parser.add_argument('-t', '--template-type', help='template type or path to generated prompt file',
                        default='a', type=str)
    return parser


def main():
    argument_parser = argparse.ArgumentParser(description='Train RelBERT.')
    argument_parser = config(argument_parser)
    opt = argument_parser.parse_args()

    # logging
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')

    trainer = relbert.Trainer(
        model=opt.model,
        max_length=opt.max_length,
        mode=opt.mode,
        data=opt.data,
        n_sample=opt.n_sample,
        template_type=opt.template_type,
        softmax_loss=opt.softmax_loss,
        in_batch_negative=opt.in_batch_negative,
        parent_contrast=opt.parent_contrast,
        mse_margin=opt.mse_margin,
        epoch=opt.epoch,
        export=opt.export,
        batch=opt.batch,
        lr=opt.lr,
        lr_decay=opt.lr_decay,
        lr_warmup=opt.lr_warmup,
        weight_decay=opt.weight_decay,
        optimizer=opt.optimizer,
        momentum=opt.momentum,
        fp16=opt.fp16,
        random_seed=opt.random_seed,
        cache_dir=opt.cache_dir)

    # add file handler
    logger = logging.getLogger()
    file_handler = logging.FileHandler('{}/training.log'.format(trainer.checkpoint_dir))
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    logger.addHandler(file_handler)

    trainer.train(num_workers=opt.num_workers, epoch_save=opt.epoch_save)


if __name__ == '__main__':
    main()
