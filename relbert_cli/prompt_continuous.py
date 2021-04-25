""" Optimize prompt for RelBERT. """
import argparse
import logging
import relbert


def config(parser):
    parser.add_argument('-e', '--epoch', help='training epochs', default=3, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.00001, type=float)
    parser.add_argument('-s', '--softmax-loss', help='softmax loss', action='store_true')
    parser.add_argument('--lr-decay', help='linear decay of learning rate after warmup', action='store_true')
    parser.add_argument("--lr-warmup", help="linear warmup of lr", default=100, type=int)
    parser.add_argument("--weight-decay", help="l2 penalty for weight decay", default=0, type=float)
    parser.add_argument('--optimizer', help='optimizer `adam`/`adamax`/`adam`', default='adam', type=str)
    parser.add_argument("--momentum", help="sgd momentum", default=0.9, type=float)
    parser.add_argument('--pseudo-token', help='pseudo token', default='<prompt>', type=str)
    # prompt
    parser.add_argument('--n-trigger-i', help='trigger number', default=3, type=int)
    parser.add_argument('--n-trigger-b', help='trigger number', default=1, type=int)
    parser.add_argument('--n-trigger-e', help='trigger number', default=1, type=int)
    # optimization
    parser.add_argument('-n', '--in-batch-negative', help='in batch negative', action='store_true')
    parser.add_argument('-p', '--parent-contrast', help='hierarchical contrastive loss', action='store_true')
    parser.add_argument('--mse-margin', help='contrastive loss margin', default=1, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=16, type=int)
    parser.add_argument('--random-seed', help='random seed', default=0, type=int)
    # training environment
    parser.add_argument('--cache-dir', help='cache directory to store dataset', default=None, type=str)
    parser.add_argument('--num-workers', help='workers for dataloder', default=1, type=int)
    parser.add_argument('--debug', help='log level', action='store_true')
    parser.add_argument('--export', help='directory name', default=None, type=str)
    # language model
    parser.add_argument('-m', '--model', help='language model', default='roberta-large', type=str)
    parser.add_argument('-l', '--max-length', help='length', default=64, type=int)
    # data
    parser.add_argument('--data', help='dataset', default='semeval2012', type=str)
    parser.add_argument('--n-sample', help='sample size', default=5, type=int)
    return parser


def main():
    argument_parser = argparse.ArgumentParser(description='Optimize prompt')
    argument_parser = config(argument_parser)
    opt = argument_parser.parse_args()

    # logging
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')

    prompter = relbert.prompt.ContinuousTriggerEmbedding(
        export=opt.export,
        epoch=opt.epoch,
        momentum=opt.momentum,
        lr=opt.lr,
        lr_warmup=opt.lr_warmup,
        lr_decay=opt.lr_decay,
        weight_decay=opt.weight_decay,
        pseudo_token=opt.pseudo_token,
        n_trigger_i=opt.n_trigger_i,
        n_trigger_b=opt.n_trigger_b,
        n_trigger_e=opt.n_trigger_e,
        model=opt.model,
        max_length=opt.max_length,
        data=opt.data,
        n_sample=opt.n_sample,
        softmax_loss=opt.softmax_loss,
        in_batch_negative=opt.in_batch_negative,
        parent_contrast=opt.parent_contrast,
        mse_margin=opt.mse_margin,
        batch=opt.batch,
        random_seed=opt.random_seed,
        cache_dir=opt.cache_dir
    )
    # add file handler
    logger = logging.getLogger()
    file_handler = logging.FileHandler('{}/process.log'.format(prompter.checkpoint_dir))
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    logger.addHandler(file_handler)

    # run
    prompter.get_prompt(opt.num_workers)


if __name__ == '__main__':
    main()
