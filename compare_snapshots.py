from typing import List, Dict

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
from plotly import graph_objs as go
from plotly.subplots import make_subplots


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
        snapshot = snapshot[:, 2:]
        infrared_part = snapshot[1:, 82:]

        self.name = name
        self.wave_lengths = snapshot[0, 82:]
        self.mean_infrared = infrared_part.mean(axis=0)
        self.left_dev = infrared_part.mean(axis=0) + calculate_dev(infrared_part, mode='left')
        self.right_dev = infrared_part.mean(axis=0) + calculate_dev(infrared_part, mode='right')


def parse_group_features(dir_list: List[str], names: List[str], max_files: int) -> Dict[str, List[SnapshotMeta]]:
    group_features: Dict[str, List[SnapshotMeta]] = {name: [] for name in names}

    for dir_id, (dir_path, name) in enumerate(zip(dir_list, names)):
        features = group_features[name]
        files = list(os.listdir(dir_path))[:max_files]

        for file_id, file in enumerate(files):
            snapshot = genfromtxt(
                f'{dir_path}/{file}',
                delimiter=','
            )
            # crop x,y
            features.append(
                SnapshotMeta(name=file, snapshot=snapshot)
            )
    return group_features


def draw_snapshots(group_features: Dict[str, List[SnapshotMeta]]):
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
    fig.write_html("comparison.html")


def draw_files(dir_list: List[str], names: List[str], max_files: int):
    group_features = parse_group_features(dir_list=dir_list, names=names, max_files=max_files)
    draw_snapshots(group_features)


if __name__ == '__main__':
    draw_files(
        dir_list=['csv/control/gala-control-bp-1_000', 'csv/phytophthora/gala-phytophthora-bp-1_000'],
        names=['health', 'phytophtora'],
        max_files=10
    )
