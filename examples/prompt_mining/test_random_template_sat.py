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
    "The <subj> has a long tail, while the <obj> has a short tail.",
    "Chisels are of two types, <subj> and <obj> chisels.",
    "<obj> oshibori are used in summer, and <subj> oshibori in winter.",
    "There are two <subj> vowels, /i/ and /u/, and one <obj> vowel /a/.",
    "The opposite of <subj> chaining is <obj> chaining.",
    "The opposite of <obj> chaining is <subj> chaining.",
    "One side is for <subj> use and the other side is for <obj> use.",
    "<subj> — Valery Vladimirovich, <obj> — Galina Ivanovna.",
    'The vast majority are of the <subj> and <obj> "tuxedo" variety.',
    "The <subj> to <obj> ratio is between 1:1.26 and 1:1.46."
]

path = '{}/template_diff.mined_template.csv'.format(EXPORT_DIR)
if not os.path.exists(path):
    template_diff = []
    for i in custom_template:
        template_diff += evaluate_analogy(relbert_ckpt=MODEL, batch_size=128, custom_template=i)
    pd.DataFrame(template_diff).to_csv(path)



