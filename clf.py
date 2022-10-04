from typing import List, Dict

import matplotlib.pyplot as plt

from sklearn import preprocessing, tree, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pandas as pd


def clf_visualize(clf, x_keys: List[str], class_labels: List[str]):
    if isinstance(clf, DecisionTreeClassifier):
        fig = plt.figure(figsize=(25, 20))
        tree.plot_tree(clf, feature_names=x_keys, class_names=class_labels, filled=True)
        fig.savefig("decision_tree.png")
    elif isinstance(clf, LogisticRegression):
        pass  # TODO implement


def clf_build(features_df: pd.DataFrame, x_keys: List[str], y_key: str, method_name='lr') -> Dict[str, float]:
    x_all = features_df[x_keys]
    y_all = features_df[y_key]

    scaler = preprocessing.StandardScaler().fit(x_all)

    x_train, x_test, y_train, y_test = train_test_split(scaler.transform(x_all), y_all, test_size=0.33, random_state=42)

    if method_name == 'lr':
        clf = LogisticRegression(random_state=16)
    elif method_name == 'dt':
        clf = DecisionTreeClassifier(random_state=16)
    elif method_name == 'svc':
        clf = SVC(random_state=16)
    else:
        raise Exception(f"Undefined clf method = {method_name}")

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    str_to_int = lambda arr: [1 if el == 'health' else 2 for el in arr]

    return {
        'accuracy': metrics.accuracy_score(y_true=y_test, y_pred=clf.predict(x_test)).round(2),

        'f1_health': metrics.f1_score(y_true=y_test, y_pred=clf.predict(x_test), pos_label="health").round(2),
        'f1_phyto': metrics.f1_score(y_true=y_test, y_pred=clf.predict(x_test), pos_label="phyto").round(2),

        'train_confusion': metrics.confusion_matrix(y_pred=y_train_pred, y_true=y_train, labels=["health", "phyto"]),
        'train_auc': metrics.auc(
            *metrics.roc_curve(y_score=str_to_int(y_train_pred), y_true=str_to_int(y_train), pos_label=2)[:2]
        ).__round__(2),

        'test_confusion': metrics.confusion_matrix(y_pred=y_test_pred, y_true=y_test, labels=["health", "phyto"]),
        'test_auc': metrics.auc(
            *metrics.roc_curve(y_score=str_to_int(y_test_pred), y_true=str_to_int(y_test), pos_label=2)[:2]
        ).__round__(2),

        'cross_val': cross_val_score(clf, scaler.transform(x_all), y_all, cv=5).round(2),
        'clf': clf
    }
