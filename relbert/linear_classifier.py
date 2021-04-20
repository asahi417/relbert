"""
You may want to pip install sklearn first
"""
from sklearn import svm

validation_dataX = [[0, 0], [1, 1]]
validation_datay = [0, 1]
validation_dataclf = svm.SVC()
clf.fit(X, y)

