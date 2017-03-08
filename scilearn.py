#
# hello machine learning
#
from sklearn import datasets
from sklearn import svm

# data
iris = datasets.load_iris()
digits = datasets.load_digits()

# train
clf = svm.SVC(gamma=0.001, C=100.0)
clf.fit(digits.data, digits.target)

# predict
print(clf.predict(digits.data[9]))
clf.fit(digits.data[:-1], digits.target[:-1])
