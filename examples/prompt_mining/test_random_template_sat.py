import os
from glob import glob
import pandas as pd
from relbert.evaluator import evaluate_classification, evaluate_analogy


MODEL = 'asahi417/relbert-roberta-large'
custom_template = "c": "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
out = evaluate_analogy(relbert_ckpt=MODEL, batch_size=128, custom_template=custom_template)


if __name__ == '__main__':
    main()
