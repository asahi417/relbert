import json
import os
from glob import glob
from itertools import combinations
import pandas as pd


analogy = {
"semeval2012_relational_similarity/validation": 0.7848101265822784,
"scan/test": 0.2592821782178218,
"sat_full/test": 0.5989304812834224,
"sat/test": 0.6083086053412463,
"u2/test": 0.5964912280701754,
"u4/test": 0.5740740740740741,
"google/test": 0.892,
"bats/test": 0.7031684269038355,
"t_rex_relational_similarity/test": 0.6666666666666666,
"conceptnet_relational_similarity/test": 0.3976510067114094,
"nell_relational_similarity/test": 0.62,
"scan/validation": 0.25842696629213485,
"sat/validation": 0.5135135135135135,
"u2/validation": 0.4583333333333333,
"u4/validation": 0.6458333333333334,
"google/validation": 0.96,
"bats/validation": 0.7738693467336684,
"t_rex_relational_similarity/validation": 0.2661290322580645,
"conceptnet_relational_similarity/validation": 0.32823741007194246,
"nell_relational_similarity/validation": 0.575
}

classification ={
"lexical_relation_classification/BLESS": {
"test/accuracy": 0.8998041283712521,
"test/f1_macro": 0.896201243435411,
"test/f1_micro": 0.8998041283712521,
"test/p_macro": 0.8876829436591316,
"test/p_micro": 0.8998041283712521,
"test/r_macro": 0.9054007585142311,
"test/r_micro": 0.8998041283712521
},
"lexical_relation_classification/CogALexV": {
"test/accuracy": 0.8370892018779342,
"test/f1_macro": 0.6583174043371445,
"test/f1_micro": 0.8370892018779342,
"test/p_macro": 0.6822907887970884,
"test/p_micro": 0.8370892018779342,
"test/r_macro": 0.6384370436284232,
"test/r_micro": 0.8370892018779342
},
"lexical_relation_classification/EVALution": {
"test/accuracy": 0.6419284940411701,
"test/f1_macro": 0.6294309369547718,
"test/f1_micro": 0.6419284940411701,
"test/p_macro": 0.6360186480100325,
"test/p_micro": 0.6419284940411701,
"test/r_macro": 0.6300178037199379,
"test/r_micro": 0.6419284940411701
},
"lexical_relation_classification/K&H+N": {
"test/accuracy": 0.9396953467343674,
"test/f1_macro": 0.8459283973092365,
"test/f1_micro": 0.9396953467343674,
"test/p_macro": 0.8614600859106621,
"test/p_micro": 0.9396953467343674,
"test/r_macro": 0.8351465630922283,
"test/r_micro": 0.9396953467343674
},
"lexical_relation_classification/ROOT09": {
"test/accuracy": 0.8815418364149169,
"test/f1_macro": 0.879329189992711,
"test/f1_micro": 0.8815418364149169,
"test/p_macro": 0.8763389203201842,
"test/p_micro": 0.8815418364149169,
"test/r_macro": 0.882560877928503,
"test/r_micro": 0.8815418364149169
}
}
if not os.path.exists("result.csv"):
    analogy = {os.path.dirname(k): round(v * 100, 1) for k, v in analogy.items() if k.endswith("test")}
    analogy.update({os.path.basename(k): round(v['test/f1_macro'] * 100, 1) for k, v in classification.items()})
    analogy["random_seed"] = "0"
    output = [analogy]
    for i in sorted(glob("relbert_output/ckpt/random_seed/*")):
        random_seed = os.path.basename(i)
        accuracy = []
        for x in glob(f"{i}/*/analogy.forward.json"):
            with open(x) as f:
                a = json.load(f)["semeval2012_relational_similarity/validation"]
            accuracy.append({"accuracy": a, "model": os.path.dirname(x)})
        best_model = sorted(accuracy, key=lambda _x: _x["accuracy"], reverse=True)[0]['model']
        print(random_seed, best_model)
        with open(f"{best_model}/analogy.forward.json") as f:
            analogy = json.load(f)
            analogy = {os.path.dirname(k): round(v * 100, 1) for k, v in analogy.items() if k.endswith("test")}
        with open(f"{best_model}/classification.json") as f:
            classification = json.load(f)
            analogy.update({os.path.basename(k): round(v['test/f1_macro'] * 100, 1) for k, v in classification.items()})
        analogy["random_seed"] = random_seed
        output.append(analogy)
    df = pd.DataFrame(output)
    df.index = df.pop("random_seed")
    df = df.T
    df.pop("2")
    df.T.std()
    df.to_csv("result.csv")
else:
    df = pd.read_csv("result.csv", index_col=0)

std_list = []
for c in combinations(df.columns, 5):
    _df = df[list(c)]
    _tmp = _df.T.std().to_dict()
    _tmp['sum'] = sum(_tmp.values())
    _tmp['ind1'] = sum(i > 1 for i in _tmp.values())
    _tmp['pairs'] = "/".join(c)
    std_list.append(_tmp)
df_std = pd.DataFrame(std_list)
df_std.sort_values(by='ind1').head(4).T
df_std.sort_values(by='sum').head(4).T
df_std.sort_values(by=['t_rex_relational_similarity']).head(4).T
df_std.sort_values(by=['sat_full']).head(1).T
df_std.sort_values(by=['u2']).head(1).T
df_std['ind2'] = df_std['sat_full'] + df_std['t_rex_relational_similarity'] + df_std['u2']
df_std.sort_values(by=['ind2']).head(4).T
target = df_std.iloc[26]
# print(df.to_latex(escape=False))