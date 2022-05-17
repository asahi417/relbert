from itertools import chain
from relbert import get_training_data

pairs = get_training_data()
for k in pairs.keys():
    print(k, len(pairs[k][0]), len(pairs[k][1]))
print(pairs)
# tmp = list(chain(*[p + n for p, n in pairs.values()]))
# print(len(tmp))
# print(tmp)
# print(len([p + n for p, n in pairs.values()]))
# print(list(chain(*[p + n for p, n in pairs.values()])))
# pairs = [pair for s, pair in chain(*[[p, n] for p, n in pairs.values()])]
# print(pairs)

