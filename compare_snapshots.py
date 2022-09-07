from typing import List, Dict

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd


def draw_klebs_np(ax, snapshot: np.ndarray, title: str):
    # for i in range(0, len(snapshot) - 1, 10):
    #     plt.plot(snapshot[0], snapshot[i + 1])
    ax.plot(snapshot[0], snapshot[1:].mean(axis=0))
    ax.set_xlim(400, 900)
    ax.set_ylim(0, 10_000)
    ax.set_title(title)


def calculate_dev(arr, mode='left') -> np.ndarray:
    residuals = arr - arr.mean(axis=0)
    ch_num = residuals.shape[1]
    dev_arr = np.zeros(ch_num)
    if mode == 'left':
        for ch_id in range(ch_num):
            res_in_ch = residuals[:, ch_id]
            dev_arr[ch_id] = res_in_ch[res_in_ch < 0].mean()
    elif mode == 'right':
        for ch_id in range(ch_num):
            res_in_ch = residuals[:, ch_id]
            dev_arr[ch_id] = res_in_ch[res_in_ch > 0].mean()

    return dev_arr


class SnapshotMeta:
    def __init__(self, name: str, snapshot: np.ndarray):
        # crop x,y
        snapshot = snapshot[:, 2:]
        # crop infrared part
        infrared_part = snapshot[1:, 82:]

        self.name = name
        self.wave_lengths = snapshot[0, 82:]
        self.mean_infrared = infrared_part.mean(axis=0)
        self.left_dev = infrared_part.mean(axis=0) + calculate_dev(infrared_part, mode='left')
        self.right_dev = infrared_part.mean(axis=0) + calculate_dev(infrared_part, mode='right')


def parse_classes(classes_dict: Dict[str, List[str]], max_files_in_dir: int) -> Dict[str, List[SnapshotMeta]]:
    classes_features: Dict[str, List[SnapshotMeta]] = {class_name: [] for class_name in classes_dict.keys()}
    for class_name, class_dirs in classes_dict.items():
        features = classes_features[class_name]
        for dir_path in class_dirs:
            files = list(os.listdir(dir_path))[:max_files_in_dir]
            for file_id, file in enumerate(files):
                snapshot = genfromtxt(
                    f'{dir_path}/{file}',
                    delimiter=','
                )
                features.append(
                    SnapshotMeta(name=file, snapshot=snapshot)
                )
    return classes_features


def draw_snapshots_as_reflectance(group_features: Dict[str, List[SnapshotMeta]]):
    max_snapshots_in_class = max([len(snapshots) for snapshots in group_features.values()])
    fig = make_subplots(
        rows=max_snapshots_in_class,
        cols=len(group_features)
    )
    colors = [('0', '180', '0'), ('180', '0', '0')]
    for col_i, ((group_key, group_list), color_tup) in enumerate(zip(group_features.items(), colors)):
        for row_i, snapshot_meta in enumerate(group_list):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta
            fig.add_trace(
                go.Scatter(
                    name=group_key,
                    legendgroup=group_key,
                    showlegend=row_i == 0,
                    x=snapshot_meta.wave_lengths,
                    y=snapshot_meta.mean_infrared,
                    line=dict(color=f"rgba({','.join(color_tup)}, 100)")
                ),
                row=row_i + 1, col=col_i + 1
            )
            fig.add_trace(
                go.Scatter(
                    name=group_key,
                    legendgroup=group_key,
                    showlegend=row_i == 0,
                    # x, then x reversed
                    x=[*snapshot_meta.wave_lengths, *snapshot_meta.wave_lengths[::-1]],
                    mode="markers+lines",
                    fill='toself',
                    line=dict(color=f"rgba({','.join(color_tup)}, 0)"),
                    # upper, then lower reversed
                    y=[*snapshot_meta.left_dev, *snapshot_meta.right_dev[::-1]]
                ),
                row=row_i + 1, col=col_i + 1
            )
    fig.update_layout(height=max_snapshots_in_class * 200, width=1800, title_text="classes reflectance comparison")
    fig.write_html("comparison_by_reflectance.html")


def get_features_df(group_features: Dict[str, List[SnapshotMeta]]) -> pd.DataFrame:
    features_dict = {'f1': [], 'f2': [], 'class': []}

    for col_i, (group_key, group_list) in enumerate(group_features.items()):
        for row_i, snapshot_meta in enumerate(group_list):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta
            features_dict['f1'].append(snapshot_meta.mean_infrared.sum())
            features_dict['f2'].append(snapshot_meta.right_dev.sum() - snapshot_meta.left_dev.sum())
            features_dict['class'].append(group_key)

    features_df = pd.DataFrame(features_dict)
    features_df['f1'] /= features_df['f1'].max()
    features_df['f2'] /= features_df['f2'].max()
    return features_df


def draw_snapshots_as_features(features_df: pd.DataFrame, colors: List[str]):
    fig = go.Figure()

    for class_name, color in zip(list(features_df['class'].unique()), colors):
        fig.add_trace(
            go.Scatter(
                name=class_name,
                legendgroup=class_name,
                mode='markers',
                x=features_df[features_df['class'] == class_name]['f1'],
                y=features_df[features_df['class'] == class_name]['f2'],
                marker=dict(
                    color=color,
                    size=20,
                    line=dict(
                        color='Black',
                        width=2
                    )
                ),
            )
        )

    fig.update_layout(height=1280, width=1800, title_text="classes reflectance comparison")
    fig.write_html("comparison_by_features.html")


def draw_files(classes_dict: Dict[str, List[str]], max_files_in_dir: int):
    classes_features_dict = parse_classes(classes_dict=classes_dict, max_files_in_dir=max_files_in_dir)
    draw_snapshots_as_reflectance(classes_features_dict)

    features_df = get_features_df(group_features=classes_features_dict)
    draw_snapshots_as_features(features_df=features_df, colors=['green', 'red'])


if __name__ == '__main__':
    draw_files(
        classes_dict={
            'health': [
                'csv/control/gala-control-bp-1_000',
                'csv/control/gala-control-bp-2_000',
                # 'csv/control/gala-control-bp-3_000',
                # 'csv/control/gala-control-bp-4_000',
            ],
            'phyto1': [
                'csv/phytophthora/gala-phytophthora-bp-1_000',
                'csv/phytophthora/gala-phytophthora-bp-5-1_000',
            ],
            # 'phyto2': [
            #     'csv/phytophthora/gala-phytophthora-bp-2_000',
            #     'csv/phytophthora/gala-phytophthora-bp-6-2_000',
            # ],
            # 'phyto3': [
            #     'csv/phytophthora/gala-phytophthora-bp-2_000',
            #     'csv/phytophthora/gala-phytophthora-bp-6-2_000',
            # ],

        },
        max_files_in_dir=10
    )
