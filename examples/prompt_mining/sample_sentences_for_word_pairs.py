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

    # wiki_dump
    dataset = load_dataset("wikipedia", '20200501.en')
    title, text = dataset.data['train']

    # word pairs
    all_positive, all_negative, relation_structure = get_training_data()
    all_word_pairs = list(chain(*(all_positive.values())))

    # filter the corpus
    path_filtered_corpus = '{}/filtered_wiki.txt'.format(export_dir)
    text = filter_text(text, all_word_pairs, path_filtered_corpus)



{'sentence': 'This New Deal-era mural was designed by the Italian artist Attilio Pusterla and painted by him and a team of artists working under his direction from 1934 and 1936, under sponsorship from the Federal Art Project of the Works Project Administration.', 'word_pairs': [['artist', 'art']]}
{'sentence': 'He was the minister of Bukka-devaraya of the Yadava Dynasty of Karnataka, his younger brother was Sayana, and Bhogantha, father was Mayana and Mother Srimati.', 'word_pairs': [['father', 'mother']]}
{'sentence': '\n\nA 2018 article by Popular Science suggests that "Dark mode is easier on the eyes and battery" and displaying white on full brightness uses roughly six times as much power as pure black on a Google Pixel, which has an OLED display.', 'word_pairs': [['black', 'white']]}
{'sentence': '\n\nIssues with the web\n\nSome argue that a color scheme with light text on a dark background is easier to read on the screen, because the lower overall brightness causes less eyestrain.', 'word_pairs': [['dark', 'light']]}
{'sentence': 'This South Stream pipeline would extend under the Black Sea to Bulgaria with a south fork to Italy and a north fork to Hungary.', 'word_pairs': [['north', 'south']]}


