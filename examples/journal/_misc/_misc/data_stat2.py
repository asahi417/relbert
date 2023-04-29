from statistics import mean
from itertools import chain
from datasets import load_dataset
import pandas as pd


labels = {
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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

x = load_dataset("relbert/semeval2012_relational_similarity", split="train")
tmp = {labels[int(k)]: [str(_x['positives'][0]) for _x in x if _x['relation_type'].startswith(k)] for k in set([i.split("/")[0] for i in x['relation_type']])}
df = pd.DataFrame([{"Relation": k, "Examples": ", ".join(v[:3])} for k, v in tmp.items()])
df.index = df.pop("Relation")
print(df.to_latex())
