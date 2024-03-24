from typing import List, Dict
import numpy as np
import pandas as pd

from drawing import draw_hp_glasses
from experiments import *
from snapshots_processing import SnapshotMeta, BandData, BANDS_DICT, parse_classes

EXP_DIR = Path('rust_2024')
EXP_DIR.mkdir(exist_ok=True)

CLASSES_DICT = {
    # **POTATO_OLD,
    # **POTATO_NEW,

    # **WHEAT_ALL,
    # **WHEAT_ALL_FILTERED,

    # **WHEAT_ALL_CLEAR_EXP,
    # **WHEAT_ALL_JUSTIFIED_EXP
    # **COCHLE_ALL_EXP_DETAILED,
    **LEAF_RUST_23_24_ALL_EXP_DETAILED
}

MAX_FILES_IN_DIR = 200


def draw_klebs_np(ax, snapshot: np.ndarray, title: str):
    # for i in range(0, len(snapshot) - 1, 10):
    #     plt.plot(snapshot[0], snapshot[i + 1])
    ax.plot(snapshot[0], snapshot[1:].mean(axis=0))
    ax.set_xlim(400, 900)
    ax.set_ylim(0, 10_000)
    ax.set_title(title)


def get_features_df(group_features: Dict[str, List[SnapshotMeta]]) -> pd.DataFrame:
    features_list = []

    for col_i, (class_name, class_snapshots) in enumerate(group_features.items()):
        for row_i, snapshot_meta in enumerate(class_snapshots):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta
            print(f'getting features for {snapshot_meta.name}')
            features_dict = snapshot_meta.get_features_dict()
            features_dict['class'] = class_name
            if 'health' in class_name or 'control' in class_name:
                features_dict['class_generalized'] = 'health'
            else:
                features_dict['class_generalized'] = 'disease'

            features_list.append(features_dict)

    features_df = pd.DataFrame(features_list)

    # normalize features
    # for key in features_df.keys():
    #     if key != 'class':
    #         features_df[key] /= features_df[key].max()

    return features_df


def draw_files(classes_features_dict: Dict[str, List[SnapshotMeta]], features_df: pd.DataFrame):
    for band_name in BANDS_DICT.keys():
        Path(f"{EXP_DIR}/{band_name}").mkdir(exist_ok=True, parents=True)

    # draw hyperspectral glasses for every band
    for band_name in BANDS_DICT.keys():
        for i in range(4):
            step = 2
            i_st = i * step
            i_fin = i * step + step
            draw_hp_glasses(
                all_classes=[classes_features_dict[key][i_st:i_fin] for key in classes_features_dict.keys()],
                classes_names=[key for key in classes_features_dict.keys()],
                bname=band_name,
                res_path=f'{EXP_DIR}/{band_name}/hp_glasses_for_snapshots[{i_st},{i_fin}].html'
            )

    # draw channels and pixels curves
    # draw_snapshots_as_reflectance(classes_features_dict,
    #                               res_path=f'{RES_DIR}/class_comparison_by_agg_in_channels.html',
    #                               x_range=(0, 900), y_range=(0, 10_000), mode='ch')
    # draw_snapshots_as_reflectance(classes_features_dict,
    #                               res_path=f'{RES_DIR}/classes_comparison_by_agg_in_pixels.html',
    #                               x_range=(0, 200), y_range=(8_000, 10_000), mode='px')

    # draw snapshots in features space
    # draw_snapshots_in_features_space(features_df=features_df, res_dir=RES_DIR)
    # draw_snapshots_in_all_paired_features_space(features_df=features_df, res_dir=RES_DIR)


def main():
    classes_features_dict = parse_classes(
        classes_dict=CLASSES_DICT,
        max_files_in_dir=MAX_FILES_IN_DIR
    )

    features_df = get_features_df(group_features=classes_features_dict)

    # draw_files(classes_features_dict=classes_features_dict, features_df=features_df)
    print(f"saving to {EXP_DIR}/features.csv")
    features_df.to_csv(f"{EXP_DIR}/features.csv", index=False)


if __name__ == '__main__':
    main()
