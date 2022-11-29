from typing import List

import pandas as pd

from drawing import draw_hp_glasses
from experiments import WHEAT_ALL_CLEAR_EXP, DYNAMIC_CHECK
from snapshots_processing import parse_classes, SnapshotMeta, WLS
import matplotlib.pyplot as plt
import numpy as np

CLASSES_DICT = {
    **DYNAMIC_CHECK
}

MAX_FILES_IN_DIR = 1000

classes_features_dict = parse_classes(
    classes_dict=CLASSES_DICT,
    max_files_in_dir=MAX_FILES_IN_DIR
)

draw_hp_glasses(
    all_classes=[classes_features_dict[key][0:2] for key in classes_features_dict.keys()],
    classes_names=[key for key in classes_features_dict.keys()],
    bname='YRI',
    res_path=f'yellow_rust.html'
)


def agg_curve(snapshots: List[SnapshotMeta]):
    # max_blue_pixels = np.array([snap.bands['all'].band_data[:, 0].max() for snap in snapshots])
    # df = pd.DataFrame({'max_blue_pixels'})
    # df.plot.hist(bins=12, alpha=0.5)
    # plt.title()
    # return np.mean([snap.bands['all'].band_data.mean(axis=0) for snap in snapshots], axis=0)
    return np.mean([snap.bands['all'].band_data.std(axis=0) for snap in snapshots], axis=0)


# for class_name, color in zip(['health2', 'puccinia phyto2'], ['#96E637', '#FF9999']):
#     class_curve = agg_curve(classes_features_dict[class_name])
#     plt.plot(WLS, class_curve, label=class_name, color=color)
#
# plt.legend(loc="lower right")
# plt.title('exp 2 56')
# plt.show()
#
# for class_name, color in zip(['health3', 'puccinia phyto3'], ['#96E637', '#FF9999']):
#     class_curve = agg_curve(classes_features_dict[class_name])
#     plt.plot(WLS, class_curve, label=class_name, color=color)
#
# plt.legend(loc="lower right")
# plt.title('exp 3')
# plt.show()


# fig, axes = plt.subplots(ncols=len(DYNAMIC_CHECK), nrows=1,
#                          figsize=(len(DYNAMIC_CHECK) * 9, 6))
# for class_name, ax in zip(DYNAMIC_CHECK.keys(), axes):
#     class_curve = agg_curve(classes_features_dict[class_name])
#     ax.plot(WLS, class_curve)
#     ax.set_title(class_name)
#     ax.set_ylim(4000, 11_000)
#
# plt.tight_layout()
# plt.legend(loc="lower right")
# plt.show()


# days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
days = [4, 5, 6, 7, 8]
# days = [0, 2, 3, 4, 5, 6, 7, 8]

fig, axes = plt.subplots(ncols=len(days), nrows=1, figsize=(len(days) * 9, 6))
for day, ax in zip(days, axes):
    health_curve = agg_curve(classes_features_dict[f'health day {day}'])
    puccina_curve = agg_curve(classes_features_dict[f'puccinia day {day}'])
    ax.plot(WLS, health_curve, color='#96E637', label='health')
    ax.plot(WLS, puccina_curve, color='#FF9999', label='puccina')
    ax.legend(loc="lower right")
    ax.set_title(f'day {day}')
    # ax.set_ylim(4000, 11_000)
    ax.set_ylim(0, 1800)

plt.tight_layout()

plt.show()
