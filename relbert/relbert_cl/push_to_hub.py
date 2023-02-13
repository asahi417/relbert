""" Push Models to Modelhub"""
import os
import argparse
import shutil
import logging
import json
from os.path import join as pj
from distutils.dir_util import copy_tree

from huggingface_hub import create_repo
from relbert.relbert_cl.readme_template import get_readme

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
gitattribute = """*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
"""


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
    parser.add_argument('--use-auth-token', help='Huggingface transformers argument of `use_auth_token`', action='store_true')
    opt = parser.parse_args()

    assert os.path.exists(pj(opt.model_checkpoint, "pytorch_model.bin"))
    logging.info(f"Upload {opt.model_checkpoint} to {opt.model_alias}")
    create_repo(repo_id=f"{opt.organization}/{opt.model_alias}", exist_ok=True, repo_type="model")
    if os.path.exists(opt.model_alias):
        shutil.rmtree(opt.model_alias)
    os.system(f"git clone https://huggingface.co/{opt.organization}/{opt.model_alias}")

    # url = create_repo(f"{opt.organization}/{opt.model_alias}", exist_ok=True)
    # args = {"use_auth_token": opt.use_auth_token, "repo_url": url, "organization": opt.organization}
    # model = RelBERT(opt.model_checkpoint)
    # assert model.is_trained
    # model_ = model.model.module if model.parallel else model.model
    # model_.push_to_hub(opt.model_alias, **args)
    # model_.config.push_to_hub(opt.model_alias, **args)
    # model.tokenizer.push_to_hub(opt.model_alias, **args)

    # config
    with open(pj(opt.model_checkpoint, "finetuning_config.json")) as f:
        trainer_config = json.load(f)
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
    with open(f"{opt.model_alias}/.gitattributes", 'w') as f:
        f.write(gitattribute)
    os.system(f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
    shutil.rmtree(opt.model_alias)  # clean up the cloned repo