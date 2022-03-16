"""
python -m spacy download en_core_web_sm
"""
import os
import json
import re
from itertools import chain
from tqdm import tqdm

import spacy
import numpy as np
from datasets import load_dataset

from relbert.data import get_training_data
from relbert import PPL
from relbert.lm import custom_prompter

splitter = spacy.load('en_core_web_sm')
export_dir = './cache'
os.makedirs(export_dir, exist_ok=True)


def split_sentence(document: str):
    return [str(i) for i in splitter(document).sents]


def filter_text(corpus, word_pairs, path):
    """ filter corpus so that it contains only word pairs. """
    print('Filtering text without the word pairs\n')
    if not os.path.exists(path):
        with open(path, 'w') as f:
            for single_text in tqdm(corpus):
                for single_sentence in split_sentence(str(single_text)):
                    tokens = single_sentence.lower().split(' ')
                    in_word_pairs = [w for w in word_pairs if w[0] in tokens and w[1] in tokens]
                    if len(in_word_pairs) == 0:
                        continue
                    f.write(json.dumps({'sentence': single_sentence, 'word_pairs': in_word_pairs}) + '\n')
    with open(path) as f:
        output = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
    print('\t * finish filtering: {} documents --> {} sentences'.format(len(corpus), len(output)))
    return output


if __name__ == '__main__':
    ###################
    # collecting data #
    ###################
    path_corpus = '{}/filtered_wiki.txt'.format(export_dir)
    # word pairs
    all_positive, all_negative, relation_structure = get_training_data()
    all_word_pairs = list(chain(*(all_positive.values())))

    if not os.path.exists(path_corpus):
        # wiki_dump
        dataset = load_dataset("wikipedia", '20200501.en')
        title, text = dataset.data['train']

        # filter the corpus
        text = filter_text(text, all_word_pairs, path_corpus)

    #############
    # filtering #
    #############
    path_filtered_corpus = '{}/filtered_wiki.cleaned.txt'.format(export_dir)
    if not os.path.exists(path_filtered_corpus):
        with open(path_corpus) as f_reader:
            full_json_list = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]
        # remove stopword
        stop = ['|', '=', '+', '(', ')', '…', '%']
        full_json_list = [i for i in full_json_list if not any(s in i['sentence'] for s in stop)]
        # remove sentence which contains '\n'
        full_json_list = [i for i in full_json_list if '\n' not in i['sentence']]
        # remove too long sentence
        full_json_list = [i for i in full_json_list if len(i['sentence']) < 64]
        # remove noisy pairs
        noisy_pairs = [['in', 'out'], ['out', 'in'], ['north', 'south'], ['south', 'north']]
        full_json_list = [i for i in full_json_list if len([p for p in i['word_pairs'] if p not in noisy_pairs]) > 0]
        # if the two word of the pair is upper case, remove since it could be a part of named entity.
        full_json_list_new = []
        for i in full_json_list:
            flag = True
            for a, b in i['word_pairs']:
                n_a = i['sentence'].lower().find(a)
                n_b = i['sentence'].lower().find(b)
                if n_a == -1 or n_b == -1:
                    flag = False
                    break
                if i['sentence'][n_a:n_a + 1].isupper() and i['sentence'][n_b:n_b + 1].isupper():
                    flag = False
                    break
            if flag:
                i['sentence'] = re.sub(r'\A\s+', '', i['sentence'])
                i['sentence'] = re.sub(r'\s+', ' ', i['sentence'])
                full_json_list_new.append(i)
        full_json_list = full_json_list_new
        # remove incomplete sentence
        full_json_list = [i for i in full_json_list if i['sentence'][0].isupper()]
        with open(path_filtered_corpus, 'w') as f_writer:
            f_writer.write('\n'.join([json.dumps(i) for i in full_json_list]))

    #############################
    # create template candidate #
    #############################
    path_template_candidate = '{}/filtered_wiki.template.jsonl'.format(export_dir)
    if not os.path.exists(path_template_candidate):
        with open(path_filtered_corpus) as f_reader:
            full_json_list = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]

        template_candid = []
        template_list = []
        noisy_pairs = [['female', 'male']]
        for i in full_json_list:
            for a, b in i['word_pairs']:
                if [a, b] in noisy_pairs:
                    continue
                template = i['sentence']

                if all(t[0].isupper() for t in template.split(' ') if t not in ['<subj>', '<obj>']):
                    continue

                if template.endswith('"'):
                    continue

                if a in b:
                    n_b = template.lower().find(b)
                    template = template[:n_b] + '<obj>' + template[n_b + len(b):]
                    n_a = template.lower().find(a)
                    template = template[:n_a] + '<subj>' + template[n_a + len(a):]
                else:
                    n_a = template.lower().find(a)
                    template = template[:n_a] + '<subj>' + template[n_a + len(a):]
                    n_b = template.lower().find(b)
                    template = template[:n_b] + '<obj>' + template[n_b + len(b):]
                assert '<subj>' in template and '<obj>' in template, template
                if '<subj> <obj>' in template or '<obj> <subj>' in template:
                    continue
                if template in template_list:
                    continue
                template_candid.append({'template': template, 'word_pair': [a, b], 'sentence': i['sentence']})
                template_list.append(template)
        with open(path_template_candidate, 'w') as f_writer:
            f_writer.write('\n'.join([json.dumps(i) for i in template_candid]))

        all_types = ['{}-{}'.format(*i['word_pair']) for i in template_candid]
        key, cnt = np.unique(all_types, return_counts=True)
        freq = sorted(list(zip(key.tolist(), cnt.tolist())), key=lambda x: x[1], reverse=True)

    #############################
    # create template candidate #
    #############################
    path_template_scores = '{}/template.score.jsonl'.format(export_dir)
    if not os.path.exists(path_template_scores):
        BATCH = int(os.getenv('BATCH', 128))
        all_scores = {}
        scorer = PPL('roberta-large', max_length=64)
        with open(path_template_candidate) as f_reader:
            template_candid = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]
        # template_candid = [i['sentence'] for i in template_candid]
        for i in tqdm(template_candid):
            prompts = [custom_prompter(p, i['template']) for p in all_word_pairs]
            out = scorer.get_perplexity(prompts, batch_size=BATCH)
            i['scores'] = {'score': out, 'prompt': prompts}

        with open(path_template_scores, 'w') as f_writer:
            f_writer.write('\n'.join([json.dumps(i) for i in template_candid]))


