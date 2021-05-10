""" Ablation study: Train model without hypernym """
import os
import logging
from copy import deepcopy
from itertools import chain

import pandas as pd
import relbert
from relbert.data import get_lexical_relation_data
from relbert.evaluator import evaluate_classification

os.makedirs('relbert_output/ablation_study/exclusion_test', exist_ok=True)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
file_handler = logging.FileHandler('relbert_output/ablation_study/exclusion_test/log.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
logger.addHandler(file_handler)
shared_relation = {
    'Random': ['random', 'RANDOM', 'false'],
    'Meronym': ['PartOf', 'PART_OF', 'mero', 'MadeOf'],
    'Event': ['event'],
    'Substance Meronym': ['HasA'],
    'Antonym': ['ANT', 'Antonym'],
    'Synonym': ['SYN', 'Synonym'],
    'Hypernym': ['HYPER', 'hyper', 'hypo', 'IsA'],
    'Co-hypornym': ['COORD', 'coord', 'sibl'],
    'Attribute': ['attri', 'HasProperty']
}

##################
# Get Data Stats #
##################
export = 'relbert_output/ablation_study/exclusion_test/data_stats.csv'
if not os.path.exists(export):

    def freq(_list, prefix=None):
        def _get(_x):
            if prefix:
                return _x[prefix]
            return _x

        f_dict = {}
        for e in _list:
            if _get(e) in f_dict:
                f_dict[_get(e)] += 1
            else:
                f_dict[_get(e)] = 1
        return f_dict


    semeval_relations = {
        1: "Class Inclusion",  # Hypernym
        2: "Part-Whole",  # Meronym, Substance Meronym
        3: "Similar",  # Synonym, Co-hypornym
        4: "Contrast",  # Antonym
        5: "Attribute",  # Attribute, Event
        6: "Non Attribute",
        7: "Case Relation",
        8: "Cause-Purpose",
        9: "Space-Time",
        10: "Representation"
    }

    data_freq = {}
    data = get_lexical_relation_data()
    for k, v in data.items():
        label = {v: k for k, v in v['label'].items()}
        data_freq[k] = {label[k]: v for k, v in freq(v['test']['y']).items()}
    relations_in_train = ['Meronym', 'Antonym', 'Synonym', 'Attribute', 'Hypernym', 'Co-hypornym', 'Substance Meronym']

    data_freq_ = deepcopy(data_freq)
    for k, v in data_freq.items():
        for _k, _v in v.items():
            for __k, __v in shared_relation.items():
                if _k in __v:
                    data_freq_[k][__k] = data_freq_[k].pop(_k)

    df = pd.DataFrame(data_freq_)
    df.to_csv(export)

##########################
# Model without Hypernym #
##########################

# os.makedirs(export, exist_ok=True)
export = 'relbert_output/ablation_study/exclusion_test/ckpt'
if not os.path.exists(export):
    trainer = relbert.Trainer(
        model='roberta-large',
        template_type='c',
        epoch=2,
        export=export,
        exclude_relation="Class Inclusion"
    )
    trainer.train()

full_result = []
target_relation = list(chain(*list(shared_relation.values())))
full_result += evaluate_classification(relbert_ckpt='relbert_output/ablation_study/exclusion_test/ckpt/epoch_2',
                                       target_relation=target_relation)
full_result += evaluate_classification(relbert_ckpt='relbert_output/ckpt/roberta_custom_c/epoch_2',
                                       target_relation=target_relation)

full_result_new = []
for x in full_result:
    i = deepcopy(x)
    for k in i.keys():
        if 'accuracy/test/' in k:
            _k = k.replace('accuracy/test/', '')
            for __k, __v in shared_relation.items():
                if _k in __v:
                    x[k.replace(_k, __k)] = x.pop(k)
    full_result_new.append(x)
os.makedirs('relbert_output/eval/summary', exist_ok=True)
pd.DataFrame(full_result_new).to_csv('relbert_output/eval/summary/ablation_study.exclusion_test.csv')
