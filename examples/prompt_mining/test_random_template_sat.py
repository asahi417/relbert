import os
import pandas as pd
from relbert.evaluator import evaluate_analogy

MODEL = 'asahi417/relbert-roberta-large'
EXPORT_DIR = 'template_experiment'
os.makedirs(EXPORT_DIR, exist_ok=True)

# test different template
path = '{}/template_diff.csv'.format(EXPORT_DIR)
if not os.path.exists(path):
    template_diff = []
    for template_type in ['a', 'b', 'c', 'd', 'e']:
        template_diff += evaluate_analogy(relbert_ckpt=MODEL, batch_size=128, template_type=template_type)
    pd.DataFrame(template_diff).to_csv(path)

# test random template
custom_template = [
    "Today, I finally discovered the relation between <subj> and <obj> : <mask>"]



