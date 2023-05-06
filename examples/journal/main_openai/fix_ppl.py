import json
from glob import glob
from math import exp, log


for _file in glob("results/scores/*"):
    print(_file)
    with open(_file) as f:
        tmp = json.load(f)
        new = []
        for j, i in tmp:
            new.append([j, [exp(-log(_i)) for _i in i]])
    with open(_file, "w") as f:
        json.dump(new, f)

