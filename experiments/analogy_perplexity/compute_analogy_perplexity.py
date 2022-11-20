import argparse
import logging
import os
import json
from tqdm import tqdm
from datasets import load_dataset
from lm_perplexity import PPL


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
templates = {
        'is-to-what': "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>",  # to-what
        'is-to-as': "<subj-a> is to <obj-a> as <subj-b> is to <obj-b>",  # to-as
        'rel-same': 'The relation between <subj-a> and <obj-a> is the same as the relation between <subj-b> and <obj-b>',  # rel-same
        'what-is-to': 'what <subj-a> is to <obj-a>, <subj-b> is to <obj-b>',  # what-to
        'she-to-as': 'She explained to him that <subj-a> is to <obj-a> as <subj-b> is to <obj-b>.',  # she-as
        'as-what-same': 'As I explained earlier, what <subj-a> is to <obj-a> is essentially the same as what <subj-b> is to <obj-b>.'  # as-what
    }


def prompting_relation(relation_words, template_type: str = 'is-to-what'):
    """ to convert a SAT style analogy set into a natural sentence with a template """



    def check_position(text, positions, tokens):
        for p, t in zip(positions, tokens):
            assert text[p[0]: p[1]] == t, '{} != {}'.format(text[p[0]: p[1]], t)

    assert template_type in templates.keys(), 'choose one from {}'.format(templates.keys())
    template = templates[template_type]
    subject_a, object_a, subject_b, object_b = relation_words
    position = []
    for i, m in zip(['<subj-a>', '<obj-a>', '<subj-b>', '<obj-b>'], [subject_a, object_a, subject_b, object_b]):
        position += [[len(template.split(i)[0]), len(template.split(i)[0]) + len(m)]]
        template = template.replace(i, m)
    check_position(template, position, [subject_a, object_a, subject_b, object_b])
    return template


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model', help='language model', required=True, type=str)
    parser.add_argument('-d', '--data', help='analogy dataset (sat/bats/u2/u4/google)', required=True, type=str)
    parser.add_argument('-e', '--export-file', help='export file', required=True, type=str)
    parser.add_argument('-p', '--prompt', help=f'prompt type: {templates.keys()}', required=True, type=str)
    parser.add_argument('--is-causal', help='', action='store_true')
    opt = parser.parse_args()

    # initialize
    logging.info(f'Computing perplexity with {opt.model} on {opt.data}')
    scorer = PPL(opt.model, is_causal=opt.is_causal)
    dataset = load_dataset("relbert/analogy_questions", opt.data)
    os.makedirs(os.path.dirname(opt.export_file), exist_ok=True)

    output = {}
    for _split in dataset:
        logging.info(f"* split: {_split}")
        ppl_list = []
        for data in tqdm(dataset[_split]):
            candidates = [data['stem'] + i for i in data['choice']]
            texts = [prompting_relation(i) for i in candidates]
            ppl = scorer.get_perplexity(texts)
            ppl_list.append(
                {"ppl": ppl, "texts": texts, "stem": data['stem'], "choice": data['choice'], "answer": data['answer']}
            )
        output[_split] = ppl_list

    with open(opt.export_file, 'w') as f:
        json.dump(output, f)
    logging.info(f'file saved at {opt.export_file}')
