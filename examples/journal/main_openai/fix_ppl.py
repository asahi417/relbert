import json
from glob import glob
from math import exp, log


for _file in glob("results/scores/*"):
    with open(_file) as f:
        tmp = json.load(f)
        new = []
        for j, i in tmp:
            new.append([j, [exp(-log(_i)) for _i in i]])

    break
