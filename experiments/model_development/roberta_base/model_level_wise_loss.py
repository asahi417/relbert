import json
import logging
import os
import shutil
from glob import glob
from distutils.dir_util import copy_tree

from huggingface_hub import create_repo
from relbert import RelBERT
from relbert.evaluation import evaluate_classification, evaluate_analogy, evaluate_relation_mapping, evaluate_validation_loss
from relbert.relbert_cl.readme_template import get_readme

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
                    trainer_config = json.load(f)
                    trainer_config['data_level'] = _level
                with open(f"{new_ckpt}/trainer_config.json", 'w') as f:
                    json.dump(trainer_config, f)

                classification = evaluate_classification(relbert_ckpt=new_ckpt, batch_size=batch)
                with open(f"{new_ckpt}/classification.json", "w") as f:
                    json.dump(classification, f)
                analogy = evaluate_analogy(relbert_ckpt=new_ckpt, batch_size=batch, max_length=max_length)
                with open(f"{new_ckpt}/analogy.json", "w") as f:
                    json.dump(analogy, f)
                mean_accuracy, _, perms_full = evaluate_relation_mapping(relbert_ckpt=new_ckpt, batch_size=batch, cache_embedding_dir="embeddings")
                relation_mapping = {"accuracy": mean_accuracy, "prediction": perms_full}
                with open(f"{new_ckpt}/relation_mapping.json", "w") as f:
                    json.dump(relation_mapping, f)

                # push to model hub
                model_alias = f"{language_model}-semeval2012-v4-{aggregate}-prompt-{prompt}-nce-{seed}-{_level.replace('_', '-')}"
                url = create_repo(f"relbert/model_alias", exist_ok=True)
                args = {"use_auth_token": False, "repo_url": url, "organization": "relbert"}
                model = RelBERT(new_ckpt)
                assert model.is_trained
                if model.parallel:
                    model_ = model.model.module
                else:
                    model_ = model.model
                model_.push_to_hub(model_alias, **args)
                model_.config.push_to_hub(model_alias, **args)
                model.tokenizer.push_to_hub(model_alias, **args)

                readme = get_readme(
                    model_name=f"relbert/{model_alias}",
                    metric_classification=classification,
                    metric_analogy=analogy,
                    metric_relation_mapping=relation_mapping,
                    config=trainer_config,
                )
                with open(f"{new_ckpt}/README.md", 'w') as f:
                    f.write(readme)
                copy_tree(new_ckpt, model_alias)
                os.system(f"cd {model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
                shutil.rmtree(model_alias)  # clean up the cloned repo

print("SKIPPED CKPT")
print(skipped)

