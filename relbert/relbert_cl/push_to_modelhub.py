""" Push Models to Modelhub"""
import os
import argparse
import shutil
import logging
from os.path import join as pj
from distutils.dir_util import copy_tree

from relbert import RelBERT

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Push to Model hub')
    parser.add_argument('-m', '--model-checkpoint', required=True, type=str)
    parser.add_argument('-a', '--model-alias', required=True, type=str)
    parser.add_argument('-o', '--organization', default='relbert', type=str)
    opt = parser.parse_args()

    assert os.path.exists(pj(opt.model_checkpoint, "pytorch_model.bin"))
    logging.info(f"Upload {opt.model_checkpoint} to {opt.model_alias}")
    model = RelBERT(opt.model_checkpoint)
    assert model.is_trained
    if model.parallel:
        model_ = model.model.module
    else:
        model_ = model.model

    model_.push_to_hub(opt.model_alias, organization=opt.organization)
    model_.config.push_to_hub(opt.model_alias, organization=opt.organization)
    model.tokenizer.push_to_hub(opt.model_alias, organization=opt.organization)

    # upload remaining files
    copy_tree(opt.model_checkpoint, opt.model_alias)
    os.system(f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
    shutil.rmtree(opt.model_alias)  # clean up the cloned repo
