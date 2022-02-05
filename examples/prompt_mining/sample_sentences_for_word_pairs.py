"""
python -m spacy download en_core_web_sm
"""
import os
import json
from itertools import chain
from tqdm import tqdm
import spacy
from datasets import load_dataset

from relbert.data import get_training_data

splitter = spacy.load('en_core_web_sm')
export_dir = './examples/prompt_mining/data'
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
                    tokens = single_sentence.split(' ')
                    in_word_pairs = [w for w in word_pairs if w[0] in tokens and w[1] in tokens]
                    if len(in_word_pairs) == 0:
                        continue
                    f.write(json.dumps({'sentence': single_sentence, 'word_pairs': in_word_pairs}) + '\n')
    with open(path) as f:
        output = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
    print('\t * finish filtering: {} documents --> {} sentences'.format(len(corpus), len(output)))
    return output


if __name__ == '__main__':

    # wiki_dump
    dataset = load_dataset("wikipedia", '20200501.en')
    title, text = dataset.data['train']

    # word pairs
    all_positive, all_negative, relation_structure = get_training_data()
    all_word_pairs = list(chain(*(all_positive.values())))

    # filter the corpus
    path_filtered_corpus = '{}/filtered_wiki.txt'.format(export_dir)
    text = filter_text(text, all_word_pairs, path_filtered_corpus)





