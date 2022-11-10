import json
import logging
import os
from glob import glob
from distutils.dir_util import copy_tree

from relbert.evaluation import evaluate_classification, evaluate_analogy, evaluate_relation_mapping, evaluate_validation_loss

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


root_dir = 'relbert_output'
language_model = 'roberta-base'
data = 'semeval2012_relational_similarity_v4'
language = 'en'
loss = 'nce'
batch = 512
max_length = 64
target_split = 'validation'
os.makedirs(f"{root_dir}/level_wise_loss", exist_ok=True)
skipped = []
for _level in ['child', 'child_prototypical', 'parent']:
    output_dir = f'{root_dir}/models/semeval2012-v4-{_level}'
    os.makedirs(output_dir, exist_ok=True)
    for aggregate in ['average', 'mask']:
        for prompt in ['a', 'b', 'c', 'd', 'e']:
            for seed in range(3):
                relbert_ckpt = glob(f'{root_dir}/models/semeval2012-v4.{prompt}.nce_logout.{aggregate}.{language_model}.*.{seed}')
                if len(relbert_ckpt) != 1:
                    skipped.append(f'{root_dir}/models/semeval2012-v4.{prompt}.nce_logout.{aggregate}.{language_model}.*.{seed}')
                    print(f'\tskip `{root_dir}/models/semeval2012-v4.{prompt}.nce_logout.{aggregate}.{language_model}.*.{seed}`')
                    continue
                relbert_ckpt = relbert_ckpt[0]
                epoch_level = []
                for epoch in range(1, 16):
                    path = f"{root_dir}/level_wise_loss/{_level}.{aggregate}.{prompt}.{seed}.{epoch}.json"
                    result = None
                    if os.path.exists(path):
                        with open(path) as f:
                            tmp_result = json.load(f)
                            if 'loss' in tmp_result and "relation_level" in tmp_result:
                                result = tmp_result
                    if result is None:
                        result = evaluate_validation_loss(
                            validation_data="relbert/semeval2012_relational_similarity_v4",
                            relbert_ckpt=f"{relbert_ckpt}/epoch_{epoch}",
                            batch_size=batch,
                            max_length=max_length,
                            split=target_split,
                            relation_level=_level
                        )
                        with open(path, 'w') as f:
                            json.dump(result, f)
                    epoch_level.append(result['loss'])
                # create new model ckpt
                best_epoch = epoch_level.index(max(epoch_level)) + 1
                new_ckpt = f"{output_dir}/{aggregate}.{prompt}.{seed}"
                copy_tree(f"{relbert_ckpt}/epoch_{best_epoch}", new_ckpt)
                with open(f"{new_ckpt}/trainer_config.json", 'r') as f:
                    tmp = json.load(f)
                    tmp['data_level'] = _level
                with open(f"{new_ckpt}/trainer_config.json", 'w') as f:
                    json.dump(tmp, f)

                result = evaluate_classification(relbert_ckpt=new_ckpt, batch_size=batch)
                with open(f"{new_ckpt}/classification.json", "w") as f:
                    json.dump(result, f)
                result = evaluate_analogy(relbert_ckpt=new_ckpt, batch_size=batch, max_length=max_length)
                with open(f"{new_ckpt}/analogy.json", "w") as f:
                    json.dump(result, f)
                mean_accuracy, _, perms_full = evaluate_relation_mapping(relbert_ckpt=new_ckpt, batch_size=batch, cache_embedding_dir="embeddings")
                result = {"accuracy": mean_accuracy, "prediction": perms_full}
                with open(f"{new_ckpt}/relation_mapping.json", "w") as f:
                    json.dump(result, f)

print(skipped)

