from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype

from clf import clf_build, cfs
from drawing import draw_hp_glasses, draw_points_tsne
from experiments import *
from snapshots_processing import SnapshotMeta, BANDS_DICT

EXP_DIR = Path('rust_november_january_detailed')
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


def get_statistics_grouped_by_key_df(features_df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    statistics_list = []
    for group_val in features_df[group_key].unique():
        class_df = features_df[features_df[group_key] == group_val]
        curr_statistics = [
            {
                'group': group_val,
                'feature': feature,
                'mean': class_df[feature].mean(),
                'std': class_df[feature].std()
            }
            for feature in class_df.keys() if is_numeric_dtype(features_df[feature])
        ]

        statistics_list.extend(curr_statistics)

    return pd.DataFrame(statistics_list)


def _get_top_tf_features(top_n: int) -> List[str]:
    return cfs(
        corr_df=features_corr_df,
        features=[tf_feature for tf_feature in features_df.keys() if 'tf_gr' in tf_feature],
        target_key='class_generalized_num',
        top_n=top_n
    )


def get_potato_rows(health_days: List[int], phyto_days: List[int], subset: int):
    if subset == 1:
        assert all([health_day in [1, 2, 3, 4] for health_day in health_days])
        assert all([phyto_day in [1, 2, 3, 4] for phyto_day in phyto_days])

        health_ids = [i for i, dir in enumerate(list(features_df['dir']))
                      if int(str(dir)[-5]) in health_days and 'control' in dir and 'singles' not in dir]

        phyto_ids = [i for i, dir in enumerate(list(features_df['dir']))
                     if int(str(dir)[-5]) in phyto_days and 'phyto' in dir and 'singles' not in dir]

        return pd.concat([features_df.iloc[health_ids], features_df.iloc[phyto_ids]])
    elif subset == 2:
        assert all([health_day in [0, 1, 3, 4, 5, 6, 7] for health_day in health_days])
        assert all([phyto_day in [0, 1, 3, 4, 5, 6, 7] for phyto_day in phyto_days])

        health_ids = [i for i, dir in enumerate(list(features_df['dir']))
                      if int(str(dir)[-5]) in health_days and 'control' in dir and 'singles' in dir]

        phyto_ids = [i for i, dir in enumerate(list(features_df['dir']))
                     if int(str(dir)[-5]) in phyto_days and 'phyto' in dir and 'singles' in dir]

        return pd.concat([features_df.iloc[health_ids], features_df.iloc[phyto_ids]])
    else:
        raise Exception(f"Unexpected subset = {subset}")


if __name__ == '__main__':
    features_df = pd.read_csv(f"{EXP_DIR}/features.csv")
    features_df.loc[:, 'class_generalized_num'] = [
        0 if ('health' in class_generalized or 'control' in class_generalized) else 1
        for class_generalized in list(features_df['class_generalized'])
    ]

    stat_df = get_statistics_grouped_by_key_df(features_df=features_df, group_key='class')
    stat_df[stat_df['feature'] == 'hNDVI']

    features_corr_df = features_df[[key for key in features_df.keys()]].corr(numeric_only=True)
    features_corr_df_by_classes = {class_name: features_df[features_df['class'] == class_name]
                                   for class_name in features_df['class'].unique()}

    features_corr_by_classes = {
        class_name: class_df[[key for key in class_df.keys()]].corr(numeric_only=True)
        for class_name, class_df in features_corr_df_by_classes.items()
    }

    clf_build(
        # fit_df=pd.concat([features_df.iloc[800:1000], features_df.iloc[1300:1500]]),
        # eval_df=pd.concat([features_df.iloc[1600:1800], features_df.iloc[2000:2200]]),
        # eval_df=pd.concat([features_df.iloc[0:200], features_df.iloc[400:600]]),
        # fit_df=pd.concat([features_df.iloc[600:800], features_df.iloc[900:1100]]),
        # fit_df=pd.concat([features_df.iloc[0:200], features_df.iloc[600:800]]),
        # eval_df=pd.concat([features_df.iloc[1000:1200], features_df.iloc[1500:1700]]),
        eval_df=pd.concat([
            features_df[features_df['class'] == 'health1'],
            features_df[features_df['class'] == 'cochle2']
        ]),
        fit_df=pd.concat([
            features_df[features_df['class'] == 'health1'],
            features_df[features_df['class'] == 'cochle3']
        ]),
        # fit_df=get_potato_rows(health_days=[1, 3, 4], phyto_days=[1, 3, 4], subset=1),
        # eval_df=get_potato_rows(health_days=[5], phyto_days=[5], subset=2),

        # features_df=features_df.iloc[[i for i, name in enumerate(list(features_df['dir']))
        #                               if int(name[-1]) in [4, 5, 6, 7]]],
        # features_df.iloc[[i for i, name in enumerate(list(features_df['dir']))
        #                   if '000' == name[-3:] and 'day' not in name and int(name[-5]) in [4, 5, 6, 7]]],
        # features_df = features_df.iloc[[i for i, name in enumerate(list(features_df['dir']))
        #                                 if 'day' in name and int(name.split('day')[1][0]) in [4, 5, 6, 7]]],
        x_keys=[
            *[
                f"{_range}_{pred}"
                for pred in [
                    'all_pixels_mean',
                    # 'all_pixels_std',
                    # 'dev_agg_in_pixels',
                    # 'dev_agg_in_channels',  # good one for all bands
                    # 'too_low_pxs_mean', 'too_high_pxs_mean',
                    # 'cl_all_het', 'cl_low_het', 'cl_high_het', 'cl_high_part', 'cl_low_part',
                ]
                # for range in BANDS_DICT.keys()
                # for _range in [str(wl) for wl in range(514, 550, 4)]
                for _range in [wl for wl in np.arange(450, 871, 4)]
                # for _range in [762, 650, 470, 766, 466, 706, 502, 718, 854, 722, 714]  # lr
                # for _range in [502, 466, 598, 718, 534, 766, 694, 650, 866, 602, 858]  # svm

            ],
            *[
                # 'ARI', 'BGI', 'BRI', 'CRI1', 'CRI2', 'CSI1', 'CSI2', 'CUR', 'gNDVI', 'hNDVI',
                # 'PRI', 'PHRI', 'NDVI', 'MSR', 'TVI', 'SIPI', 'NPCI', 'ARI', 'GI', 'TCARI', 'PSRI', 'RVSI', 'NRI', 'YRI'
                # 'NPCI'
            ],
            *[
                # tf_feature for tf_feature in features_df.keys() if 'tf_gr' in tf_feature
                # *list(_get_top_tf_features(top_n=20))
            ]
        ],
        y_key='class_generalized',
        # method_name='lr',
        method_name='svc',
        # clf_args=dict(kernel='rbf', C=3.),
        # dec_analyze=True,
        # clf_pretrained=clf_results['clf'],
        scaler_fit_on_all=False
    )

    wl_keys = [key for key in features_df.keys() if 'all_pixels_mean' in key]

    common_draw_args = dict(s=300, edgecolors='black', linewidth=1)
    plt.figure(figsize=(12, 8))
    draw_points_tsne(
        # pt_groups=[
        #     features_df[features_df['class'] == 'control_day_4'][wl_keys].to_numpy(),
        #     features_df[features_df['class'] == 'control_day_5'][wl_keys].to_numpy(),
        #     features_df[features_df['class'] == 'cochle_day_4'][wl_keys].to_numpy(),
        #     features_df[features_df['class'] == 'cochle_day_5'][wl_keys].to_numpy()
        # ],
        # groups_draw_args=[
        #     dict(color='#5DBB63', label='control day 4', **common_draw_args),
        #     dict(color='#AEF359', label='control day 5', **common_draw_args),
        #     dict(color='#D0312D', label='cochle day 4', **common_draw_args),
        #     dict(color='#900D09', label='cochle day 5', **common_draw_args),
        # ]
        pt_groups=[
            features_df[features_df['class'] == 'exp=3,group=control,day=1'][wl_keys].to_numpy(),
            features_df[features_df['class'] == 'exp=3,group=control,day=2'][wl_keys].to_numpy(),
            features_df[features_df['class'] == 'exp=3,group=control,day=3'][wl_keys].to_numpy(),
            features_df[features_df['class'] == 'exp=3,group=control,day=4'][wl_keys].to_numpy(),
            features_df[features_df['class'] == 'exp=3,group=control,day=5'][wl_keys].to_numpy(),

            features_df[features_df['class'] == 'exp=3,group=cochle,day=1'][wl_keys].to_numpy(),
            features_df[features_df['class'] == 'exp=3,group=cochle,day=2'][wl_keys].to_numpy(),
            features_df[features_df['class'] == 'exp=3,group=cochle,day=3'][wl_keys].to_numpy(),
            features_df[features_df['class'] == 'exp=3,group=cochle,day=4'][wl_keys].to_numpy(),
            features_df[features_df['class'] == 'exp=3,group=cochle,day=5'][wl_keys].to_numpy()

        ],
        groups_draw_args=[
            dict(color='#7CFF92', label='exp=3,group=control,day=1', **common_draw_args),
            dict(color='#5BFD76', label='exp=3,group=control,day=2', **common_draw_args),
            dict(color='#23F045', label='exp=3,group=control,day=3', **common_draw_args),
            dict(color='#12CE31', label='exp=3,group=control,day=4', **common_draw_args),
            dict(color='#02A91E', label='exp=3,group=control,day=5', **common_draw_args),

            dict(color='#FF6056', label='exp=3,group=cochle,day=1', **common_draw_args),
            dict(color='#EC291C', label='exp=3,group=cochle,day=2', **common_draw_args),
            dict(color='#C92115', label='exp=3,group=cochle,day=3', **common_draw_args),
            dict(color='#AF0C00', label='exp=3,group=cochle,day=4', **common_draw_args),
            dict(color='#800B03', label='exp=3,group=cochle,day=5', **common_draw_args),

        ]
    )
    plt.legend(fancybox=True, framealpha=0.5, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f'{EXP_DIR}/exp3_tsne.png')
    plt.show()
    plt.clf()
