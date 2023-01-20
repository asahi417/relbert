import os
import logging
import argparse
from relbert import evaluate_analogy

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main_analogy():
    parser = argparse.ArgumentParser(description='RelBERT evaluation on analogy question')
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('-o', '--output-file', help='export file', required=True, type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=512, type=int)
    parser.add_argument('-l', '--max-length', help='for vanilla LM', default=64, type=int)
    parser.add_argument('-d', '--data', help='target analogy', default=None, type=str)
    parser.add_argument('--aggregation-mode', help='aggregation mode (for vanilla LM)', default='average_no_mask', type=str)
    parser.add_argument('-t', '--template', help='template (for vanilla LM)', default=None, type=str)
    parser.add_argument('-c', '--classification-loss', help='softmax loss', action='store_true')
    opt = parser.parse_args()
    out = evaluate_analogy(
        relbert_ckpt=opt.model,
        max_length=opt.max_length,
        batch_size=opt.batch,
        target_analogy=opt.data,
        distance_function='cosine_similarity',
        reverse_pair=False,
        bi_direction_pair=False,
        aggregation_mode=opt.aggregation_mode,
        template=opt.template
    )
    if os.path.dirname(opt.output_file) != '':
        os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)
    with open(opt.output_file, 'w') as f:
        f.write(out)

