import os
from glob import glob
import pandas as pd
from relbert.evaluator import evaluate_classification, evaluate_analogy


MODEL = 'asahi417/relbert-roberta-large'
EXPORT_DIR = 'template_experiment'
os.makedirs(EXPORT_DIR, exist_ok=True)

# test different template
template_diffrence = {}
for template_type in ['a', 'b', 'c', 'd', 'e']:
    out = evaluate_analogy(relbert_ckpt=MODEL, batch_size=128, custom_template=custom_template, template_type=template_type)
    template_diffrence[template_type] = {
        'accuracy/full': out['accuracy/full'],
        'accuracy/test': out['accuracy/test'],
        'data': out['data'],
        'template': out['template']
    }


custom_template = "Today, I finally discovered the relation between <subj> and <obj> : <mask>"


