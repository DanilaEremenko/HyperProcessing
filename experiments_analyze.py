from typing import List

import pandas as pd

from experiments import WHEAT_ALL_CLEAR_EXP, WHEAT_ALL_JUSTIFIED_EXP, DYNAMIC_WHEAT_CHECK
from snapshots_processing import parse_classes, SnapshotMeta, WLS
import matplotlib

font = {'family': 'Times New Roman',
        'size': 28}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import numpy as np

CLASSES_DICT = {
    **DYNAMIC_WHEAT_CHECK
}

MAX_FILES_IN_DIR = 1000

classes_features_dict = parse_classes(
    classes_dict=CLASSES_DICT,
    max_files_in_dir=MAX_FILES_IN_DIR
)


# from drawing import draw_hp_glasses
# draw_hp_glasses(
#     all_classes=[classes_features_dict[key][0:2] for key in classes_features_dict.keys()],
#     classes_names=[key for key in classes_features_dict.keys()],
#     bname='all',
#     res_path=f'all.html'
# )


def agg_curve(snapshots: List[SnapshotMeta]):
    # max_blue_pixels = np.array([snap.bands['all'].band_data[:, 0].max() for snap in snapshots])
    # df = pd.DataFrame({'max_blue_pixels'})
    # df.plot.hist(bins=12, alpha=0.5)
    # plt.title()
    # return np.mean([snap.bands['all'].band_data.mean(axis=0) for snap in snapshots], axis=0)
    return np.mean([snap.bands['all'].band_data.mean(axis=0) for snap in snapshots], axis=0)


def agg_index(snapshots: List[SnapshotMeta], index_name: str):
    return np.mean([snap.get_indexes_features()[index_name] for snap in snapshots], axis=0)


classes_prefs = ['health', 'puccinia']
classes_colors = ['#96E637', '#FF9999']
assert len(classes_colors) >= len(classes_prefs)
days = sorted(list(set([int(key[-1]) for key in classes_features_dict.keys()])))
exp_pref = 'exp3'
days_postf = 'fair'

########################################################################################################################
# ---------------------------------------- draw agg curves -------------------------------------------------------------
########################################################################################################################
fig, axes = plt.subplots(ncols=len(days), nrows=1, figsize=(len(days) * 9, 6))
for day, ax in zip(days, axes):
    for class_pref, class_color in zip(classes_prefs, classes_colors):
        curr_curve = agg_curve(classes_features_dict[f'{class_pref} day {day}'])
        ax.plot(WLS, curr_curve, color=class_color, label=class_pref, linewidth=4)
    ax.legend(loc="lower right")
    ax.set_title(f'day {day}')
    ax.set_ylim(4000, 11_000)
    # ax.set_ylim(0, 1800)

plt.tight_layout()

plt.savefig(f'topic/{exp_pref}_curves_{days_postf}.png')
plt.show()

########################################################################################################################
# ---------------------------------------- draw indexes ----------------------------------------------------------------
########################################################################################################################
indexes = ['CRI1', 'CRI2', 'LCI', 'PSSRa', 'PSSRb', 'PSSRc', 'SR(Chla)', 'SR(Chlb)', 'SR(Chlb2)', 'SR(Chltot)']
fig, axes = plt.subplots(ncols=1, nrows=len(indexes), figsize=(12, len(indexes) * 3))
res_list = {'day': days}

for ax, index_name in zip(axes, indexes):
    for class_pref, class_color in zip(classes_prefs, classes_colors):
        day_indexes = [
            agg_index(classes_features_dict[f'{class_pref} day {day}'], index_name=index_name)
            for day in days
        ]
        ax.plot(days, day_indexes, color=class_color, label=class_pref, linewidth=4)
        ax.legend(loc="lower right")
        ax.set_xlabel('day')
        ax.set_ylabel(index_name)
        res_list[index_name] = day_indexes

fig.tight_layout()

plt.savefig(f'topic/{exp_pref}_indexes_by_days_{days_postf}.png')
plt.show()

index_df = pd.DataFrame(res_list)
index_df.to_csv(f'topic/{exp_pref}_indexes_by_days_{days_postf}.df', index=False)
