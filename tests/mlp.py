from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import cross_val_score


rng = np.random.default_rng(0)

N = 10000
X = rng.normal(size=(N, 10))
y = rng.binomial(n=1, p=0.5, size=N)


clf_0 = MLPClassifier(random_state=0)
clf_1 = MLPClassifier(random_state=0)
clf_2 = MLPClassifier(random_state=0)

print(cross_val_score(clf_0, X, y, cv=2, scoring='roc_auc'))
print(cross_val_score(clf_1, X, y, cv=2, scoring='roc_auc'))
print(cross_val_score(clf_2, X, y, cv=2, scoring='roc_auc'))
