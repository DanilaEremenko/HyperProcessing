from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from sklearn import preprocessing, tree, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pandas as pd


def clf_decision_analyze(clf, features: List[str], class_labels: List[str]):
    if isinstance(clf, DecisionTreeClassifier):
        fig = plt.figure(figsize=(25, 20))
        tree.plot_tree(clf, feature_names=features, class_names=class_labels, filled=True)
        res_path = "decision_tree.png"
        fig.savefig(res_path)
        print(f"Decision tree structure saved into {res_path}")
    elif isinstance(clf, LogisticRegression):
        imps = np.abs(clf.coef_.flatten())
        imps = softmax(imps)
        assert len(imps) == len(features)
        imp_dict = dict(zip(features, imps))
        imp_dict = dict(sorted(imp_dict.items(), key=lambda item: abs(item[1])))

        for key, val in imp_dict.items():
            print(f"{key}:{val}")


def clf_build(
        features_df: pd.DataFrame,
        x_keys: List[str],
        y_key: str,
        method_name='lr',
        clf_args: Optional[dict] = None,
        dec_analyze=False
) -> Dict[str, float]:
    clf_args = {} if clf_args is None else clf_args

    x_all = features_df[x_keys]
    y_all = features_df[y_key]

    scaler = preprocessing.StandardScaler().fit(x_all)

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.33, random_state=42)

    if method_name == 'lr':
        clf = make_pipeline(scaler, LogisticRegression(**clf_args, random_state=16))
    elif method_name == 'dt':
        clf = make_pipeline(scaler, DecisionTreeClassifier(**clf_args, random_state=16))
    elif method_name == 'svc':
        clf = make_pipeline(scaler, SVC(**clf_args, random_state=16))
    else:
        raise Exception(f"Undefined clf method = {method_name}")

    clf.fit(x_train, y_train)

    if dec_analyze:
        clf_decision_analyze(clf=clf[-1], features=x_keys, class_labels=['health', 'phyto'])

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
        'cross_val': cross_val_score(
            clf, x_all, y_all,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ).round(2),
        'clf': clf
    }
