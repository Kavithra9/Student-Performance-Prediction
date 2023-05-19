import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def random_forest_classifier(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))
    print("Recall:", metrics.recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", metrics.f1_score(y_test, y_pred, average='macro'))
    return clf


def logistic_regression(x_train, x_test, y_train, y_test):
    clf = LogisticRegression(random_state=42).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))
    print("Recall:", metrics.recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", metrics.f1_score(y_test, y_pred, average='macro'))
    return clf


def decision_tree(x_train, x_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=42, criterion='gini')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))
    print("Recall:", metrics.recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", metrics.f1_score(y_test, y_pred, average='macro'))
    return clf


def naive_bayes(x_train, x_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))
    print("Recall:", metrics.recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", metrics.f1_score(y_test, y_pred, average='macro'))
    return clf
