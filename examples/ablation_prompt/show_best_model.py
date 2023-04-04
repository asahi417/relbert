import json
import os
from glob import glob
import pandas as pd

template_list = {
    "template-1": "Today, I finally discovered the spaceship between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-2": "Today, I finally discovered Napoleon Bonaparte between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-3": "Today, I finally discovered football between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-4": "Today, I finally discovered Italy between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-5": "Today, I finally discovered Cardiff between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-6": "Today, I finally discovered the earth science between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-7": "Today, I finally discovered pizza between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-8": "Today, I finally discovered subway between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-9": "Today, I finally discovered ocean between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-10": "Today, I finally discovered Abraham Lincoln between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-length_1": "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "template-length_2": "Today, I finally discovered that the relation between <subj> and <obj> is <mask>",
    "template-length_3": "I discovered that the relation between <subj> and <obj> is <mask>",
    "template-length_4": "the relation between <subj> and <obj> is <mask>",
    "template-length_5": "<subj> and <obj> is <mask>"
}

if not os.path.exists("result.csv"):
    output = []
    for i in sorted(glob("relbert_output/ckpt/random_template/template*")):
        template_type = os.path.basename(i)
        accuracy = []
        for x in glob(f"{i}/*/analogy.forward.json"):
            with open(x) as f:
                a = json.load(f)["semeval2012_relational_similarity/validation"]
            accuracy.append({"accuracy": a, "model": os.path.dirname(x)})
        if len(accuracy) != 10:
            continue
        best_model = sorted(accuracy, key=lambda _x: _x["accuracy"], reverse=True)[0]['model']
        with open(f"{best_model}/analogy.forward.json") as f:
            analogy = json.load(f)
            analogy = {os.path.dirname(k): round(v * 100, 1) for k, v in analogy.items() if k.endswith("test")}
        with open(f"{best_model}/classification.json") as f:
            classification = json.load(f)
            analogy.update({os.path.basename(k): round(v['test/f1_macro'] * 100, 1) for k, v in classification.items()})
        analogy["template_type"] = template_type
        analogy["template"] = template_list[template_type].replace("<subj>", "\textbf{[h]}").replace("<obj>", "\textbf{[t]}").replace("<mask>", "\texttt{<mask>}")
        output.append(analogy)
    df = pd.DataFrame(output)
    df.index = df.pop("template_type")
    df = df.T
    df.to_csv("result.csv")
else:
    df = pd.read_csv("result.csv", index_col=0)
print(df.to_latex(escape=False))