""" Train model without hypernym """
import os
import logging
import pandas as pd
import relbert
from relbert.evaluator.classification import evaluate

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# for t, name in zip(['c',
#                     './relbert_output/prompt_files/d922/prompt.json',
#                     './relbert_output/prompt_files/c932/prompt.json'],
#                    ['Manual', 'AutoPrompt', 'P-tuning']):
for t, name in zip(['c'], ['manual']):
    export = 'examples/analysis_data/ckpt/{}'.format(name)
    if os.path.exists(export):
        continue
    trainer = relbert.Trainer(
        model='roberta-large',
        template_type=t,
        epoch=2,
        export=export,
        exclude_relation="Class Inclusion"
    )
    trainer.train()

full_result = []
for name in ['manual']:
# for name in ['Manual', 'AutoPrompt', 'P-tuning']:
    export = 'examples/analysis_data/ckpt/{}/epoch_2'.format(name)
    full_result += evaluate(relbert_ckpt=export, target_relation=['HYPER', 'hyper', 'hypo', 'IsA'])
    print(full_result[-1])

for export in ['relbert_output/ckpt/roberta_custom_c/epoch_2']:
# for export in ['relbert_output/ckpt/roberta_custom_c/epoch_2',
#                'relbert_output/ckpt/roberta_auto_d922/epoch_2',
#                'relbert_output/ckpt/roberta_auto_c932/epoch_2']:
    full_result += evaluate(relbert_ckpt=export, target_relation=['HYPER', 'hyper', 'hypo', 'IsA'])
    print(full_result[-1])

pd.DataFrame(full_result).to_csv('examples/analysis_data/result.csv')
