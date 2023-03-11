import os
import json

from relbert import evaluate_analogy

os.makedirs("results/relbert_prediction", exist_ok=True)
for t in ["scan", "sat_full", "u2", "u4", "bats", "google", "t_rex_relational_similarity", "nell_relational_similarity", "conceptnet_relational_similarity"]:
    path = f"results/relbert_prediction/{t}.json"
    if not os.path.exists(path):
        out = evaluate_analogy(
                target_analogy=t,
                relbert_ckpt="relbert/relbert-roberta-large",
                max_length=64,
                batch_size=64,
                target_analogy_split="test",
                aggregation=False)
        with open(path, "w") as f:
            json.dump(out, f)

    path = f"results/relbert_prediction_base/{t}.json"
    if not os.path.exists(path):
        out = evaluate_analogy(
                target_analogy=t,
                relbert_ckpt="relbert/relbert-roberta-base",
                max_length=64,
                batch_size=64,
                target_analogy_split="test",
                aggregation=False)
        with open(path, "w") as f:
            json.dump(out, f)
