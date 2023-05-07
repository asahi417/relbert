from random import seed, shuffle
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})  # must set in top
styles = ['o-', 'o--', 'o:', 's-', 's--', 's:', '^-', '^--', '^:', "X-", "X--", "X:"]
seed(1)
colors = list(mpl.colormaps['tab20b'].colors)
shuffle(colors)

df = pd.read_csv("result.csv", index_col=0)
df.pop("template_id")
df.pop("epoch")
df_a = df[['SAT', 'U2', 'U4', 'BATS', 'Google', 'SCAN', 'NELL', 'T-REX', 'ConceptNet']]
df_a["Average"] = df_a.mean(1).round(1)
ax = df_a.plot(grid=True, style=styles, color=colors, figsize=(8, 4))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("analogy.png", bbox_inches="tight", dpi=600)

df_c = df[['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']]
df_c["Average"] = df_c.mean(1).round(1)
ax = df_c.plot(grid=True, style=styles, color=colors, figsize=(8, 4))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("classification.png", bbox_inches="tight", dpi=600)
