""" Ablation study: Train model without hypernym """
import os
import logging
from copy import deepcopy
from itertools import chain

import pandas as pd
import relbert
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

##########################
# Model without Hypernym #
##########################
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
# load checkpoint from model hub
full_result += evaluate_classification(relbert_ckpt="asahi417/relbert_roberta_custom_c",
                                       target_relation=target_relation)
full_result += evaluate_classification(relbert_ckpt='relbert_output/ablation_study/exclusion_test/ckpt/epoch_2',
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
