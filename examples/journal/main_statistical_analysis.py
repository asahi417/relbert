import json
from glob import glob
from pprint import pprint
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


# load 1st baseline
target = {}  # dataset name: list accuracy
for i in glob("examples/journal/main_lm/results/relbert_prediction/*.json"):
    with open(i) as f:
        target.update(json.load(f))
target = {k.split("/")[0]: v for k, v in target.items()}
with open("examples/journal/main_lm/results/relbert_prediction_base/t_rex_relational_similarity.json") as f:
    target["t_rex_relational_similarity"] = json.load(f)["t_rex_relational_similarity/test"]

# load 2nd baseline
baseline = {}  # dataset name: list accuracy
with open("examples/journal/main_lm/results/fasttext_prediction/google.json") as f:
    baseline["google"] = json.load(f)["full"]
with open("examples/journal/main_lm/results/fasttext_prediction/nell_relational_similarity.json") as f:
    baseline["nell_relational_similarity"] = json.load(f)["full"]
with open("examples/journal/main_lm/results/relbert_prediction_base/scan.json") as f:
    baseline["scan"] = json.load(f)["scan/test"]
with open("examples/journal/main_lm/results/relbert_prediction_base/sat_full.json") as f:
    baseline["sat_full"] = json.load(f)["sat_full/test"]
with open("examples/journal/main_lm/results/relbert_prediction_base/u2.json") as f:
    baseline["u2"] = json.load(f)["u2/test"]
with open("examples/journal/main_lm/results/relbert_prediction_base/u4.json") as f:
    baseline["u4"] = json.load(f)["u4/test"]
with open("examples/journal/main_lm/results/relbert_prediction_base/conceptnet_relational_similarity.json") as f:
    baseline["conceptnet_relational_similarity"] = json.load(f)["conceptnet_relational_similarity/test"]
with open("examples/journal/main_lm/results/relbert_prediction/t_rex_relational_similarity.json") as f:
    baseline["t_rex_relational_similarity"] = json.load(f)["t_rex_relational_similarity/test"]
baseline["bats"] = pd.read_csv(f"examples/journal/main_lm/results/breakdown/flan-ul2_bats_None.prompt.csv")["accuracy"].values.tolist()


# t-test
ttest_result = {}
for k in baseline:
    ttest_result[k] = ttest_ind(
        np.array(baseline[k]).astype(int),
        np.array(target[k]).astype(int),
        alternative="less"
    ).pvalue < 0.05
pprint(ttest_result)

# load 3rd baseline
baseline_3 = {}  # dataset name: list accuracy
baseline_3["google"] = pd.read_csv(f"examples/journal/main_lm/results/breakdown/flan-ul2_google_None.prompt.csv")["accuracy"].values.tolist()
baseline_3["t_rex_relational_similarity"] = pd.read_csv(f"examples/journal/main_lm/results/breakdown/flan-ul2_t_rex_relational_similarity_None.prompt.csv")["accuracy"].values.tolist()

# t-test
ttest_result = {}
for k in baseline_3:
    ttest_result[k] = ttest_ind(
        np.array(baseline_3[k]).astype(int),
        np.array(target[k]).astype(int),
        alternative="less"
    ).pvalue < 0.05
pprint(ttest_result)