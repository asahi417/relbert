import json
import os
from glob import glob
from itertools import combinations
import pandas as pd


analogy_base = {
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

classification_base ={
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
analogy_base = {os.path.dirname(k): round(v * 100, 1) for k, v in analogy_base.items() if k.endswith("test")}
analogy_base.update({os.path.basename(k): round(v['test/f1_macro'] * 100, 1) for k, v in classification_base.items()})
analogy_base["negative_sample"] = 400
analogy_base["template_id"] = "a"
analogy_base["epoch"] = 8


def get_result(sample, template_id, ckpt):
    ckpt = f"relbert_output/ckpt/negative_sample_{sample}/template-{template_id}/{ckpt}"
    with open(f"{ckpt}/analogy.forward.json") as f:
        analogy = json.load(f)
        analogy = {os.path.dirname(k): round(v * 100, 1) for k, v in analogy.items() if k.endswith("test")}
    with open(f"{ckpt}/classification.json") as f:
        classification = json.load(f)
        analogy.update({os.path.basename(k): round(v['test/f1_macro'] * 100, 1) for k, v in classification.items()})
    analogy["negative_sample"] = sample
    analogy["template_id"] = template_id
    analogy["epoch"] = 10 if ckpt.endswith('model') else int(os.path.basename(ckpt).split("_")[-1])
    return analogy


if __name__ == '__main__':
    if not os.path.exists("result.csv"):
        output = [analogy_base]
        output.append(get_result(25, "a", "model"))
        output.append(get_result(50, "a", "epoch_6"))
        output.append(get_result(100, "b", "epoch_6"))
        output.append(get_result(200, "a", "epoch_8"))
        output.append(get_result(250, "a", "epoch_9"))
        output.append(get_result(150, "e", "epoch_8"))
        output.append(get_result(300, "e", "model"))
        output.append(get_result(350, "e", "epoch_9"))
        output.append(get_result(450, "a", "epoch_8"))
        output.append(get_result(500, "a", "epoch_9"))
        df = pd.DataFrame(output)
        df = df.sort_values(by="negative_sample")
        df = df[['sat_full', 'u2', 'u4', 'bats', 'google', 'scan', 'nell_relational_similarity',
                 't_rex_relational_similarity', 'conceptnet_relational_similarity',
                 'BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09', 'negative_sample', 'template_id', 'epoch']]
        df.columns = ['SAT', 'U2', 'U4', 'BATS', 'Google', 'SCAN', 'NELL', 'T-REX', 'ConceptNet',
                      'BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09', 'negative_sample', 'template_id', 'epoch']
        df.index = df.pop("negative_sample").values
        df.to_csv("result.csv")
    df = pd.read_csv("result.csv", index_col=0)
    df.pop("template_id")
    df.pop("epoch")
    df["Average (Analogy)"] = df[['SAT', 'U2', 'U4', 'BATS', 'Google', 'SCAN', 'NELL', 'T-REX', 'ConceptNet']].mean(1).round(1)
    df["Average (CLS)"] = df[['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']].mean(1).round(1)
    df = df[['SAT', 'U2', 'U4', 'BATS', 'Google', 'SCAN', 'NELL', 'T-REX', 'ConceptNet', "Average (Analogy)",
             'BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09', "Average (CLS)"]]
    df = df.T
    df = pd.DataFrame([[f"\textbf{{{k}}}" if k == max(i) else str(k) for k in i] for i in df.values], index=df.index, columns=df.columns)
    print(df.to_latex(escape=False))
