import pandas as pd
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv("result.csv", index_col=0)
df.pop("template_id")
df.pop("epoch")
corr = []
for c in df.columns:
    corr.append({"data": c, "corr": spearmanr(df[c], df.index)[0], "p": spearmanr(df[c], df.index)[1]})
corr = pd.DataFrame(corr)

print(corr)
