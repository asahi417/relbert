""" Push Models to Modelhub"""
import os
import argparse
import shutil
import logging
import json
from os.path import join as pj
from distutils.dir_util import copy_tree

from huggingface_hub import create_repo
from relbert import RelBERT
from relbert.relbert_cl.readme_template import get_readme

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def safe_json_load(_file):
    if os.path.exists(_file):
        with open(_file) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description='Push model to model hub.')
    parser.add_argument('-m', '--model-checkpoint', required=True, type=str)
    parser.add_argument('-a', '--model-alias', required=True, type=str)
    parser.add_argument('-o', '--organization', default='relbert', type=str)
    parser.add_argument('--use-auth-token', help='Huggingface transformers argument of `use_auth_token`',
                        action='store_true')
    opt = parser.parse_args()

    assert os.path.exists(pj(opt.model_checkpoint, "pytorch_model.bin"))
    logging.info(f"Upload {opt.model_checkpoint} to {opt.model_alias}")
    url = create_repo(f"{opt.organization}/{opt.model_alias}", exist_ok=True)
    args = {"use_auth_token": opt.use_auth_token, "repo_url": url, "organization": opt.organization}
    model = RelBERT(opt.model_checkpoint)
    assert model.is_trained
    model_ = model.model.module if model.parallel else model.model
    model_.push_to_hub(opt.model_alias, **args)
    model_.config.push_to_hub(opt.model_alias, **args)
    model.tokenizer.push_to_hub(opt.model_alias, **args)

    # config
    with open(pj(opt.model_checkpoint, "finetuning_config.json")) as f:
        trainer_config = json.load(f)

    # metric
    analogy = safe_json_load(pj(opt.model_checkpoint, "analogy.forward.json"))
    classification = safe_json_load(pj(opt.model_checkpoint, "classification.json"))
    relation_mapping = safe_json_load(pj(opt.model_checkpoint, "relation_mapping.json"))

    readme = get_readme(
        model_name=f"{opt.organization}/{opt.model_alias}",
        metric_classification=classification,
        metric_analogy=analogy,
        metric_relation_mapping=relation_mapping,
        config=trainer_config,
    )
    with open(pj(opt.model_checkpoint, "README.md"), 'w') as f:
        f.write(readme)

    # upload remaining files
    copy_tree(opt.model_checkpoint, opt.model_alias)
    os.system(f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
    shutil.rmtree(opt.model_alias)  # clean up the cloned repo