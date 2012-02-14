import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt

from data import libras_movement

X, Y = libras_movement()

# Shuffle the data
idxs = np.arange(len(X))
idxs = np.random.permutation(idxs)

X = X[idxs]
Y = Y[idxs]

# Split the data into two sets: 80% for training, 20% for testing
X_train, Y_train = X[:len(X) * 8 / 10], Y[:len(Y) * 8 / 10]
X_test, Y_test = X[len(X) * 8 / 10:], Y[len(Y) * 8 / 10:]

# We will be using a binary classifier to do multi class classification. We
# have two options in order to solve this problem:
#  - One against One
#  - One against All
#
# Here is implemented the One vs All version. We will combine n classifier
# (for n classes) to make a decision function.


def predict(c, X_train, Y_train, X_test, Y_test, kernel):
    """Predict"""

    num_classes = int(Y.max())

    classifiers = []
    for i in range(num_classes):
        Y_tmp = (Y_train == i + 1)
        svc = SVC(C=c, probability=True, kernel=kernel)
        svc.fit(X_train, Y_tmp)
        classifiers.append(svc)

    predictions = np.zeros((num_classes, len(X_train)))
    for i, classifier in enumerate(classifiers):
        predictions[i] = classifier.predict_proba(X_train)[:, 1]

    labels = predictions.argmax(axis=0) + 1
    true_pred_train = float((labels == Y_train).sum()) / len(labels)

    predictions = np.zeros((num_classes, len(X_test)))
    for i, classifier in enumerate(classifiers):
        predictions[i] = classifier.predict_proba(X_test)[:, 1]

    labels = predictions.argmax(axis=0) + 1
    true_pred_test = float((labels == Y_test).sum()) / len(labels)

    return true_pred_train, true_pred_test


C = [0.001, 0.005, 0.01, 0.05,
     0.1, 1, 5, 10, 50, 100, 500, 1000, 5000,
     10000, 5 * 1e4, 1e5, 5 * 1e5, 1e6, 5 * 1e6, 1e7]
kernels = ('linear', 'rbf', 'poly')
for i, kernel in enumerate(kernels):
    print "Computing for kernel %s" % kernel

    tp_train = []
    tp_test = []
    for c in C:
        print ".",
        train, test = predict(c, X_train, Y_train, X_test, Y_test, kernel)
        tp_train.append(train)
        tp_test.append(test)

    fig = plt.figure(i)
    ax = fig.add_subplot(111)
    ax.plot(C, tp_train)
    ax.plot(C, tp_test)
    ax.set_xscale('log')
    ax.legend(('Training', 'Testing'), shadow=True)
