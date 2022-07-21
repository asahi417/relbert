""" Push Models to Modelhub"""
import os
import argparse
from os.path import join as pj
from distutils.dir_util import copy_tree

import transformers


def main():
    parser = argparse.ArgumentParser(description='Push to Model hub')
    parser.add_argument('-m', '--model-checkpoint', required=True, type=str)
    parser.add_argument('-a', '--model-alias', required=True, type=str)
    opt = parser.parse_args()

    assert os.path.exists(pj(opt.model_checkpoint, "pytorch_model.bin"))
    print(f"Upload {opt.model_checkpoint} to {opt.model_alias}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(opt.model_checkpoint, local_files_only=True)
    config = transformers.AutoConfig.from_pretrained(opt.model_checkpoint, local_files_only=True)
    model = transformers.AutoModel(opt.model_checkpoint, config=config, local_files_only=True)
    model.push_to_hub(opt.model_alias)
    tokenizer.push_to_hub(opt.model_alias)
    config.push_to_hub(opt.model_alias)

    # upload remaining files
    copy_tree(f"{opt.model_checkpoint}", f"{opt.model_alias}")
    os.system(f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")