from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from sklearn import preprocessing, tree, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pandas as pd


def cfs(corr_df: pd.DataFrame, features: List[str], target_key: str, top_n: int) -> List[str]:
    """
    Correlation based feature selection method implementation
    """
    tf_features_corr_df = corr_df.loc[
        [*features, target_key],
        [*features, target_key]
    ]

    tf_features_imp_df = [
                             {
                                 'feature_name': feature_name,
                                 'corr_imp': feature_corr_vector.iloc[-1].__abs__() /
                                             feature_corr_vector.iloc[:-1].abs().mean()
                             }
                             for feature_name, feature_corr_vector in tf_features_corr_df.iterrows()
                         ][:-1]
    tf_features_imp_df = pd.DataFrame(tf_features_imp_df).sort_values('corr_imp', ascending=False).reset_index(
        drop=True)

    return tf_features_imp_df.iloc[:top_n]['feature_name']


def clf_decision_analyze(clf, features: List[str], class_labels: List[str]):
    if isinstance(clf, DecisionTreeClassifier):
        fig = plt.figure(figsize=(25, 20))
        tree.plot_tree(clf, feature_names=features, class_names=class_labels, filled=True)
        res_path = "decision_tree.png"
        fig.savefig(res_path)
        print(f"Decision tree structure saved into {res_path}")
    elif isinstance(clf, LogisticRegression) or isinstance(clf, SVC):
        imps = np.abs(clf.coef_.flatten())
        imps = softmax(imps)
        assert len(imps) == len(features)
        imp_dict = dict(zip(features, imps))
        imp_dict = dict(sorted(imp_dict.items(), key=lambda item: abs(item[1])))

        for key, val in imp_dict.items():
            print(f"{key}:{val}")


def clf_build(
        fit_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        x_keys: List[str],
        y_key: str,
        method_name='lr',
        clf_args: Optional[dict] = None,
        dec_analyze=False,
        clf_pretrained=None,
        scaler_fit_on_all=False
) -> Dict[str, float]:
    x_fit = fit_df[x_keys]
    y_fit = fit_df[y_key]

    x_eval = eval_df[x_keys]
    y_eval = eval_df[y_key]

    if scaler_fit_on_all:
        scaler = preprocessing.StandardScaler().fit(pd.concat([x_fit, x_eval]))
    else:
        scaler = preprocessing.StandardScaler().fit(x_fit)

    x_train, x_test, y_train, y_test = train_test_split(x_fit, y_fit, test_size=0.33, random_state=42)

    if clf_pretrained is None:
        if method_name == 'lr':
            if clf_args is not None:
                clf = make_pipeline(scaler, LogisticRegression(**clf_args))
            else:
                common_args = {'max_iter': [1e5], 'random_state': [16]}
                param_grid = [
                    {'solver': ['lbfgs'], 'C': [1e0, 1e1, 1e2, 1e3], 'penalty': ['l2'], **common_args},
                    # {'solver': ['sag'], 'C': [1e0, 1e1, 1e2, 1e3], 'penalty': ['l2'],**common_args},
                    # {'solver': ['saga'], 'C': [1e0, 1e1, 1e2, 1e3], 'penalty': ['elasticnet', 'l1', 'l2'],**common_args},
                ]
                clf_grid = GridSearchCV(LogisticRegression(), param_grid)
                clf_grid.fit(X=scaler.transform(x_fit), y=y_fit)
                clf = make_pipeline(scaler, clf_grid.best_estimator_)
        elif method_name == 'dt':
            clf = make_pipeline(scaler, DecisionTreeClassifier(random_state=16))
        elif method_name == 'svc':
            if clf_args is not None:
                clf = make_pipeline(scaler, SVC(**clf_args))
            else:
                common_args = {'max_iter': [int(1e5)], 'random_state': [16]}
                param_grid = [
                    {'C': [1e0, 1e1, 1e2, 1e3], 'kernel': ['linear'], **common_args},
                    {'C': [1e0, 1e1, 1e2, 1e3], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],'kernel': ['rbf', 'poly', 'sigmoid'], **common_args},
                ]
                clf_grid = GridSearchCV(SVC(), param_grid)
                clf_grid.fit(X=scaler.transform(x_fit), y=y_fit)
                clf = make_pipeline(scaler, clf_grid.best_estimator_)
        else:
            raise Exception(f"Undefined clf method = {method_name}")

        clf.fit(x_train, y_train)
    else:
        clf = clf_pretrained

    if dec_analyze:
        clf_decision_analyze(clf=clf[-1], features=x_keys, class_labels=['health', 'disease'])

    samples_dict = {
        'train': {'y_true': y_train, 'y_pred': clf.predict(x_train)},
        'test': {'y_true': y_test, 'y_pred': clf.predict(x_test)},
    }
    if eval_df is not None:
        samples_dict['eval'] = {'y_true': y_eval, 'y_pred': clf.predict(x_eval)}

    str_to_int = lambda arr: [1 if el == 'health' else 2 for el in arr]

    def get_metrics_dict(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            f'{name}_accuracy': metrics.accuracy_score(y_true=y_true, y_pred=y_pred).__round__(2),
            # f'{name}_auc': metrics.auc(*metrics.roc_curve(
            #     y_score=str_to_int(y_pred), y_true=str_to_int(y_true), pos_label=2)[:2]).__round__(2),
            f'{name}_kappa': metrics.cohen_kappa_score(y_true, y_pred).round(2),
            # f'{name}_f1_health': metrics.f1_score(y_true=y_true, y_pred=y_pred, pos_label="health").round(2),
            f'{name}_f1_disease': metrics.f1_score(y_true=y_true, y_pred=y_pred, pos_label="disease").round(2),
            f'{name}_confusion': metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=["health", "disease"])
        }

    return {
        **{
            metric_key: metric_val
            for ds_name, ds_dict in samples_dict.items()
            for metric_key, metric_val in get_metrics_dict(name=ds_name, **ds_dict).items()
        },
        'cross_val': cross_val_score(
            clf, x_fit, y_fit,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ).round(2),
        'clf': clf
    }
