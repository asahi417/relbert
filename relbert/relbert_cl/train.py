""" Train RelBERT model.
relbert-train -o relbert_output/ckpt/batch39 -b 39 -e 1
relbert-eval-analogy -m relbert_output/ckpt/batch39/model -o relbert_output/ckpt/batch39/model/analogy.json
relbert-train -o relbert_output/ckpt/batch39c -b 39 -e 1 -c
relbert-eval-analogy -m relbert_output/ckpt/batch39c/model -o relbert_output/ckpt/batch39c/model/analogy.json

relbert-train -o relbert_output/ckpt/batch32nce -b 32 -e 10 --loss nce --temperature 0.05 -r 0.00005 --num-negative 10 --num-positive 10
relbert-eval-analogy -m relbert_output/ckpt/batch64nce/epoch_1 -o relbert_output/ckpt/batch64nce/epoch_1/analogy.json
relbert-eval-analogy -m relbert_output/ckpt/batch64nce/epoch_2 -o relbert_output/ckpt/batch64nce/epoch_2/analogy.json
relbert-eval-analogy -m relbert_output/ckpt/batch64nce/epoch_3 -o relbert_output/ckpt/batch64nce/epoch_3/analogy.json
relbert-eval-analogy -m relbert_output/ckpt/batch64nce/epoch_4 -o relbert_output/ckpt/batch64nce/epoch_4/analogy.json
relbert-eval-analogy -m relbert_output/ckpt/batch64nce/epoch_5 -o relbert_output/ckpt/batch64nce/epoch_5/analogy.json
relbert-eval-analogy -m relbert_output/ckpt/batch64nce/epoch_6 -o relbert_output/ckpt/batch64nce/epoch_6/analogy.json
relbert-eval-analogy -m relbert_output/ckpt/batch64nce/epoch_7 -o relbert_output/ckpt/batch64nce/epoch_7/analogy.json
relbert-eval-analogy -m relbert_output/ckpt/batch64nce/epoch_8 -o relbert_output/ckpt/batch64nce/epoch_8/analogy.json
relbert-eval-analogy -m relbert_output/ckpt/batch64nce/epoch_9 -o relbert_output/ckpt/batch64nce/epoch_9/analogy.json
relbert-eval-analogy -m relbert_output/ckpt/batch64nce/model -o relbert_output/ckpt/batch64nce/model/analogy.json
"""
import argparse
import logging
import relbert


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
DEFAULT_TEMPLATE = "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"


def main():
    parser = argparse.ArgumentParser(description='Train RelBERT.')

    # model
    parser.add_argument('-o', '--output-dir', help='output directly', required=True, type=str)
    parser.add_argument('-t', '--template', help='template', default=DEFAULT_TEMPLATE, type=str)
    parser.add_argument('-m', '--model', help='language model', default='roberta-large', type=str)
    parser.add_argument('-l', '--max-length', help='length', default=64, type=int)

    # training
    parser.add_argument('-e', '--epoch', help='training epochs', default=1, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=64, type=int)
    parser.add_argument('-s', '--random-seed', help='random seed', default=0, type=int)
    parser.add_argument('-r', '--lr', help='learning rate', default=0.00002, type=float)
    parser.add_argument('-w', "--lr-warmup", help="linear warmup of lr", default=10, type=int)

    # others
    parser.add_argument('--aggregation-mode', help='aggregation mode', default='average_no_mask', type=str)
    parser.add_argument('--data', help='data', default='relbert/semeval2012_relational_similarity', type=str)
    parser.add_argument('--exclude-relation', help="", nargs='+', default=None, type=str)
    parser.add_argument('--split', help='', default='train', type=str)
    parser.add_argument('--loss', help='', default='triplet', type=str)
    parser.add_argument('-c', '--classification-loss', help='softmax loss', action='store_true')

    # config: triplet loss
    parser.add_argument('--mse-margin', help='contrastive loss margin', default=1, type=int)
    parser.add_argument('--temperature', help='temperature for nce', default=0.1, type=float)
    parser.add_argument('--gradient-accumulation', help='gradient accumulation', default=1, type=int)
    parser.add_argument('--num-negative', help='gradient accumulation', default=300, type=int)
    parser.add_argument('--num-positive', help='gradient accumulation', default=10, type=int)

    # misc
    parser.add_argument('--epoch-save', help='interval to save model weight', default=1, type=int)

    # logging
    opt = parser.parse_args()
    if opt.loss == 'triplet':
        loss_function_config = {'mse_margin': opt.mse_margin}
    elif opt.loss in ['nce', 'iloob']:
        loss_function_config = {'temperature': opt.temperature,
                                "gradient_accumulation": opt.gradient_accumulation,
                                "num_negative": opt.num_negative,
                                "num_positive": opt.num_positive,
                                }
    else:
        loss_function_config = {}
    trainer = relbert.Trainer(
        output_dir=opt.output_dir,
        template=opt.template,
        model=opt.model,
        max_length=opt.max_length,
        epoch=opt.epoch,
        batch=opt.batch,
        random_seed=opt.random_seed,
        lr=opt.lr,
        lr_warmup=opt.lr_warmup,
        aggregation_mode=opt.aggregation_mode,
        data=opt.data,
        exclude_relation=opt.exclude_relation,
        split=opt.split,
        loss_function=opt.loss,
        classification_loss=opt.classification_loss,
        loss_function_config=loss_function_config
    )

    trainer.train(epoch_save=opt.epoch_save)
