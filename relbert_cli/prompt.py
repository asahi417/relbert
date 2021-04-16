""" Optimize prompt for RelBERT. """
import argparse
import logging
import relbert


def config(parser):
    # prompt
    parser.add_argument('--method', help='prompting method', default='autoprompt', type=str)
    parser.add_argument('-k', '--topk', help='top k', default=5, type=int)
    parser.add_argument('--trigger-selection', help='method to select trigger', default='best', type=str)
    parser.add_argument('--n-trigger-i', help='trigger number', default=3, type=int)
    parser.add_argument('--n-trigger-b', help='trigger number', default=1, type=int)
    parser.add_argument('--n-trigger-e', help='trigger number', default=1, type=int)
    parser.add_argument('-i', '--n-iteration', help='iteration', default=25, type=int)
    parser.add_argument('--filter-label', help='remove label token', action='store_true')
    parser.add_argument('--filter-pn', help='remove proper noun', action='store_true')
    # optimization
    parser.add_argument('-n', '--in-batch-negative', help='in batch negative', action='store_true')
    parser.add_argument('-p', '--parent-contrast', help='hierarchical contrastive loss', action='store_true')
    parser.add_argument('--mse-margin', help='contrastive loss margin', default=1, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=16, type=int)
    parser.add_argument('--batch-no-grad', help='batch size', default=512, type=int)
    parser.add_argument('--random-seed', help='random seed', default=0, type=int)
    # training environment
    parser.add_argument('--cache-dir', help='cache directory to store dataset', default=None, type=str)
    parser.add_argument('--num-workers', help='workers for dataloder', default=1, type=int)
    parser.add_argument('--debug', help='log level', action='store_true')
    parser.add_argument('--export-dir', help='directory to export', default=None, type=str)
    parser.add_argument('--export-name', help='directory name', default=None, type=str)
    # language model
    parser.add_argument('-m', '--model', help='language model', default='roberta-large', type=str)
    parser.add_argument('--checkpoint-path', help='checkpoint to load', default=None, type=str)
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

    if opt.method == 'autoprompt':
        prompter = relbert.prompt.GradientTriggerSearch(
            topk=opt.topk,
            trigger_selection=opt.trigger_selection,
            n_trigger_i=opt.n_trigger_i,
            n_trigger_b=opt.n_trigger_b,
            n_trigger_e=opt.n_trigger_e,
            n_iteration=opt.n_iteration,
            filter_label=opt.filter_label,
            filter_pn=opt.filter_pn,
            model=opt.model,
            max_length=opt.max_length,
            data=opt.data,
            n_sample=opt.n_sample,
            in_batch_negative=opt.in_batch_negative,
            parent_contrast=opt.parent_contrast,
            mse_margin=opt.mse_margin,
            batch=opt.batch,
            random_seed=opt.random_seed,
            export_dir=opt.export_dir,
            export_name=opt.export_name,
            cache_dir=opt.cache_dir,
            checkpoint_path=opt.checkpoint_path
        )
    else:
        raise ValueError('unknown method: {}'.format(opt.method))

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
