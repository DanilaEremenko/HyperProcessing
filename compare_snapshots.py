from typing import List

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os


def draw_klebs_np(ax, snapshot: np.ndarray, title: str):
    # for i in range(0, len(snapshot) - 1, 10):
    #     plt.plot(snapshot[0], snapshot[i + 1])
    ax.plot(snapshot[0], snapshot[1:].mean(axis=0))
    ax.set_xlim(400, 900)
    ax.set_ylim(0, 10_000)
    ax.set_title(title)


def draw_files(dir_list: List[str], names: List[str], max_files: int):
    fig, axes = plt.subplots(nrows=max_files, ncols=len(dir_list))

    for dir_id, (dir_path, name) in enumerate(zip(dir_list, names)):
        files = list(os.listdir(dir_path))[:max_files]
        for file_id, file in enumerate(files):
            snapshot = genfromtxt(
                f'{dir_path}/{file}',
                delimiter=','
            )
            snapshot = snapshot[:, 2:]
            draw_klebs_np(ax=axes[file_id, dir_id], snapshot=snapshot, title=f'snapshot {name} {file_id}')
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    draw_files(
        dir_list=['csv/control/gala-control-bp-1_000', 'csv/phytophthora/gala-phytophthora-bp-1_000'],
        names=['health', 'phytophtora'],
        max_files=10
    )
