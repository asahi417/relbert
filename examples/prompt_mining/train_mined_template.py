import os
import logging
import subprocess
from glob import glob


def reader(_file):
    with open(_file) as f:
        return [i for i in f.read().split('\n') if len(i) > 0]


# get prompt
with open('./run.sh', 'w') as f_writer:
    prompts = {}
    for i in glob('output/template.*.*10.csv'):
        with open(i) as f:
            prompts = f.read().split('\n')
        for n, p in enumerate(prompts):
            if len(p) == 0:
                continue
            output = os.path.basename(i).replace('.csv', '')
            f_writer.write(f'relbert-train -m roberta-large -n -p -s --export ./ckpt/{output}.{n} --custom-template "{p}" \n')
