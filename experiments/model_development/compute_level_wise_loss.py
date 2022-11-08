import json
import logging
import os
from glob import glob
from relbert.evaluation import evaluate_classification, evaluate_analogy, evaluate_validation_loss, \
    evaluate_relation_mapping

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


root_dir = 'relbert_output/models'
language_model = 'roberta-base'
data = 'semeval2012_relational_similarity_v4'
language = 'en'
loss = 'nce'
batch = 512
max_length = 64
target_split = 'validation'
os.makedirs(f"relbert/semeval2012_relational_similarity_v4/level_wise_loss", exist_ok=True)
skipped = []
for _level in ['child', 'child_prototypical', 'parent']:
    for aggregate in ['average', 'mask']:
        for prompt in ['a', 'b', 'c', 'd', 'e']:
            for seed in range(3):
                relbert_ckpt = glob(f'{root_dir}/semeval2012-v4.{prompt}.nce_logout.{aggregate}.{language_model}.*.{seed}')
                if len(relbert_ckpt) != 1:
                    skipped.append(f'{root_dir}/semeval2012-v4.{prompt}.nce_logout.{aggregate}.{language_model}.*.{seed}')
                    print(f'\tskip `{root_dir}/semeval2012-v4.{prompt}.nce_logout.{aggregate}.{language_model}.*.{seed}`')
                    continue

                for epoch in range(1, 16):
                    result = evaluate_validation_loss(
                        validation_data=f"relbert/semeval2012_relational_similarity_v4",
                        relbert_ckpt=f"{relbert_ckpt[0]}/epoch_{epoch}",
                        batch_size=batch,
                        max_length=max_length,
                        split=target_split,
                        relation_level=_level
                    )
                    with open(f"relbert/semeval2012_relational_similarity_v4/level_wise_loss/{_level}.{aggregate}.{prompt}.{seed}.{epoch}.json", 'w') as f:
                        json.dump(result, f)


