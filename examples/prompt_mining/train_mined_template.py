import os
import logging
import subprocess
from glob import glob


def exe_shell(command: str, exported_file: str = None):
    """ Execute shell command """
    logging.info("execute `{}`".format(command))
    try:
        args = dict(stderr=subprocess.STDOUT, shell=True, timeout=600, universal_newlines=True)
        log = subprocess.check_output(command, **args)
        logging.info("log\n{}".format(log))
    except subprocess.CalledProcessError as exc:
        if exported_file and os.path.exists(exported_file):
            # clear possibly broken file out
            os.system('rm -rf {}'.format(exported_file))
        raise ValueError("fail to execute command `{}`:\n {}\n {}".format(command, exc.returncode, exc.output))


def reader(_file):
    with open(_file) as f:
        return [i for i in f.read().split('\n') if len(i) > 0]


# get prompt
prompts = {}
for i in glob('cache/template.*.*10.csv'):
    with open(i) as f:
        prompts = f.read().split('\n')
    for p in prompts:
        if len(p) == 0:
            continue
        output = os.path.basename(i).replace('.csv', '')
        exe_shell(f'relbert-train -m roberta-large -n -p -s --export ./ckpt/{output} --custom-template {p}')
