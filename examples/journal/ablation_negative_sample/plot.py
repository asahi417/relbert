import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("result.csv", index_col=0)
df.pop("template_id")
df.pop("epoch")
df_a = df[['SAT', 'U2', 'U4', 'BATS', 'Google', 'SCAN', 'NELL', 'T-REX', 'ConceptNet']]
df_a["Average"] = df_a.mean(1).round(1)
ax = df_a.plot()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("analogy.png", bbox_inches="tight", dpi=600)

df_c = df[['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']]
df_c["Average"] = df_c.mean(1).round(1)
ax = df_c.plot()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("classification.png", bbox_inches="tight", dpi=600)
