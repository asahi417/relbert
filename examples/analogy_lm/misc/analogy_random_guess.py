import json
from statistics import mean
from datasets import load_dataset

types = ["sat_full", "u2", "u4", "bats", "google", "t_rex_relational_similarity", "nell_relational_similarity", "conceptnet_relational_similarity"]
output = {}
for t in types:
    data = load_dataset("relbert/analogy_questions", t, split="test")
    output[t] = mean([1/len(i) for i in data['choice']])
print(json.dumps(output, indent=4))