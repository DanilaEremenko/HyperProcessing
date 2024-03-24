import os
from datetime import datetime
from pathlib import Path

POTATO_OLD = {
    'potato health 1': [
        f'csv/control/gala-control-bp-{day}_000'
        for day in [1, 2, 3, 4]
    ],
    'potato phyto1': [
        *[
            f'csv/phytophthora/gala-phytophthora-bp-{day}_000'
            for day in [1, 2, 3, 4]
        ]
    ]
}

POTATO_NEW = {
    'potato health 2': [
        f'csv/potato-singles-1/singles-1/gala-control/gala-control-{day}_000'
        for day in [0, 1, 3, 4, 5, 6]

    ],
    'potato phyto2': [
        f'csv\potato-singles-1\singles-1\gala-phytophthora\gala-phytophthora-{day}_000'
        for day in [0, 1, 3, 4, 5, 6]
    ],
}

WHEAT_ALL = {
    'health1': [
        f'csv/sep-wheat-1/wheat-control/wheat-control-{i}'
        for i in range(4, 10, 1)
    ],
    'puccinia phyto1': [
        f'csv/sep-wheat-1/wheat-puccinia/wheat-puccinia-{i}'
        for i in range(4, 10, 1)
    ],

    'health2': [
        f'csv/sep-wheat-2/wheat-control/wheat-control-{i}_000'
        for i in range(4, 10, 1)
    ],
    'puccinia phyto2': [
        f'csv/sep-wheat-2/wheat-puccinia/wheat-puccinia-{i}_000'
        for i in range(4, 10, 1)
    ],

    'health3': [
        f'csv/sep-wheat-3/wheat-control/wheat_day{i}_control_000'
        for i in range(4, 10, 1)
    ],
    'puccinia phyto3': [
        f'csv/sep-wheat-3/wheat-puccinia/wheat_day{i}_experiment_000'
        for i in range(4, 10, 1)
    ]
}

WHEAT_ALL_FILTERED = {
    'health1': [
        f'csv/sep-wheat-1/wheat-control/wheat-control-{i}'
        for i in [0, 1, 2, 3]
    ],
    'puccinia phyto1': [
        f'csv/sep-wheat-1/wheat-puccinia/wheat-puccinia-{i}'
        for i in [4, 5, 6, 7]
    ],

    'health2': [
        f'csv/sep-wheat-2/wheat-control/wheat-control-{i}_000'
        for i in [0, 2, 3, 4]
    ],
    'puccinia phyto2': [
        f'csv/sep-wheat-2/wheat-puccinia/wheat-puccinia-{i}_000'
        for i in [4, 5, 6, 7]
    ],

    'health3': [
        f'csv/sep-wheat-3/wheat-control/wheat_day{i}_control_000'
        for i in [4, 5, 6, 7]
    ],
    'puccinia phyto3': [
        f'csv/sep-wheat-3/wheat-puccinia/wheat_day{i}_experiment_000'
        for i in [4, 5, 6, 7]
    ]
}

WHEAT_ALL_CLEAR_EXP = {
    'health2': [
        f'csv/sep-wheat-2/wheat-control/wheat-control-{i}_000'
        for i in [0, 2]
    ],
    'puccinia phyto2': [
        f'csv/sep-wheat-2/wheat-puccinia/wheat-puccinia-{i}_000'
        for i in [5, 6]
    ],

    'health3': [
        f'csv/sep-wheat-3/wheat-control/wheat_day{i}_control_000'
        for i in [4, 5]
    ],
    'puccinia phyto3': [
        f'csv/sep-wheat-3/wheat-puccinia/wheat_day{i}_experiment_000'
        for i in [4, 5]
    ]
}

WHEAT_ALL_JUSTIFIED_EXP = {
    'health2': [
        f'csv/sep-wheat-2/wheat-control/wheat-control-{i}_000'
        for i in [0, 2, 0, 2, 0]
    ],
    'puccinia phyto2': [
        f'csv/sep-wheat-2/wheat-puccinia/wheat-puccinia-{i}_000'
        for i in [4, 5, 6, 7, 8]
    ],

    'health3': [
        f'csv/sep-wheat-3/wheat-control/wheat_day{i}_control_000'
        for i in [4, 5, 4, 5, 4]
    ],
    'puccinia phyto3': [
        f'csv/sep-wheat-3/wheat-puccinia/wheat_day{i}_experiment_000'
        for i in [4, 5, 6, 7, 8]
    ]
}

COCHLE_EXP = {
    'control_day_4': ['csv/new_data/cochle-control4_000', ],
    'control_day_5': ['csv/new_data/cochle-control5_000'],
    'cochle_day_4': ['csv/new_data/cochle-experiment4_000'],
    'cochle_day_5': ['csv/new_data/cochle-experiment5_000']
}

COCHLE_ALL_EXP = {
    'health1': [
        f'csv/sep-cochle-1/cochle-contr{i}_000'
        for i in [4, 5]
    ],
    'cochle1': [
        f'csv/sep-cochle-1/cochle-exp{i}_000'
        for i in [4, 5]
    ],

    'health2': [
        f'csv/sep-cochle-2/cocle2-contr{i}_000'
        for i in [1, 2, 3]
    ],
    'cochle2': [
        f'csv/sep-cochle-2/cocle2-exp{i}_000'
        for i in [1, 2, 3]
    ],

    'health3': [
        f'csv/sep-cochle-3/cocle3-contr{i}_000'
        for i in [1, 2, 3, 4, 5]
    ],
    'cochle3': [
        f'csv/sep-cochle-3/cocle3-exp{i}_000'
        for i in [1, 2, 3, 4, 5]
    ]
}

COCHLE_ALL_EXP_DETAILED = {
    **{
        f'exp=1,group=control,day={i}': [f'csv/sep-cochle-1/cochle-contr{i}_000']
        for i in [4, 5]
    },
    **{
        f'exp=1,group=cochle,day={i}': [f'csv/sep-cochle-1/cochle-exp{i}_000']
        for i in [4, 5]
    },

    **{
        f'exp=2,group=control,day={i}': [f'csv/sep-cochle-2/cocle2-contr{i}_000']
        for i in [1, 2, 3]
    },
    **{
        f'exp=2,group=cochle,day={i}': [f'csv/sep-cochle-2/cocle2-exp{i}_000']
        for i in [1, 2, 3]
    },

    **{
        f'exp=3,group=control,day={i}': [f'csv/sep-cochle-3/cocle3-contr{i}_000']
        for i in [1, 2, 3, 4, 5]
    },
    **{
        f'exp=3,group=cochle,day={i}': [f'csv/sep-cochle-3/cocle3-exp{i}_000']
        for i in [1, 2, 3, 4, 5]
    }
}


def get_key_from_path_cocle_23(path: str) -> str:
    day = int(path[-5])

    group = Path(path).name.split('_')[1][:-1]
    if group == 'contr':
        group = 'control'
    elif group == 'exp':
        group = 'cocle'
    else:
        raise Exception(f"Undefined group = {group} in {path}")

    date_str = Path(path).parent.name
    date = datetime.strptime(date_str, '%Y_%m_%d')

    exp_id = int(Path(path).name.replace('cocle', '')[0])
    assert exp_id in [1, 2, 3], exp_id
    return f'exp={exp_id},group={group},day={day}'


COCHLE_ALL_EXP_DETAILED_LAST = {
    **{
        get_key_from_path_cocle_23(path=f"{dir}/{subdir}/{group_dir}"): [f"{dir}/{subdir}/{group_dir}"]
        for exp_dir in ['csv/Leaf cochle3']
        for dir, subdirs, files in os.walk(exp_dir)
        for subdir in subdirs
        if '2023_' in subdir
        for group_dir in os.listdir(f"{dir}/{subdir}")
    }

}


def get_key_from_path_rust_23_24(path: str) -> str:
    day = int(path[-5])

    group = Path(path).name.split('_')[1][:-1]
    if group == 'contr':
        group = 'control'
    elif group == 'exp':
        group = 'rust'
    else:
        raise Exception(f"Undefined group = {group} in {path}")

    date_str = Path(path).parent.name
    date = datetime.strptime(date_str, '%Y_%m_%d')

    # return f'date={date_str},group={group},day={day}'
    if '2023' in path:
        return f'exp=2023_nov,group={group},day={day}'
    elif '2024' in path and 'jan' in path:
        return f'exp=2024_jan,group={group},day={day}'
    elif '2024' in path and 'feb' in path:
        return f'exp=2024_feb,group={group},day={day}'
    else:
        raise Exception(f"No year in path {path}")


LEAF_RUST_23_24_ALL_EXP_DETAILED = {
    **{
        get_key_from_path_rust_23_24(path=f"{dir}/{subdir}/{group_dir}"): [f"{dir}/{subdir}/{group_dir}"]
        for exp_dir in [
            'csv/Leaf rust 1 november 2023',
            'csv/Leaf rust 2 january 2024',
            'csv/Leaf rust february 2024']
        for dir, subdirs, files in os.walk(exp_dir)
        for subdir in subdirs
        if '2023_' in subdir or '2024_' in subdir
        for group_dir in os.listdir(f"{dir}/{subdir}")
    }

}
WHEAT_ALL_WORKING_EXP = {
    'health2': [
        f'csv/sep-wheat-2/wheat-control/wheat-control-{i}_000'
        for i in [0, 2]
    ],
    'puccinia phyto2': [
        f'csv/sep-wheat-2/wheat-puccinia/wheat-puccinia-{i}_000'
        for i in [5, 6]
    ],

    'health3': [
        f'csv/sep-wheat-3/wheat-control/wheat_day{i}_control_000'
        for i in [4, 5]
    ],
    'puccinia phyto3': [
        f'csv/sep-wheat-3/wheat-puccinia/wheat_day{i}_experiment_000'
        for i in [4, 5]
    ]
}

DYNAMIC_WHEAT_CHECK = {
    # **{
    #     f'puccinia day {i}': [f'csv/sep-wheat-1/wheat-puccinia/wheat-puccinia-{i}']
    #     for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # },
    # **{
    #     f'health day {i}': [f'csv/sep-wheat-1/wheat-control/wheat-control-{i}']
    #     for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # },

    # **{
    #     f'puccinia day {i}': [f'csv/sep-wheat-2/wheat-puccinia/wheat-puccinia-{i}_000']
    #     for i in [4, 5, 6, 7, 8]
    # },
    #
    # **{
    #     f'health day {pi}': [f'csv/sep-wheat-2/wheat-control/wheat-control-{hi}_000']
    #     # for hi, pi in zip([0, 2, 0, 2, 0], [4, 5, 6, 7, 8])
    #     for hi, pi in zip([0, 2, 3, 4, 5], [4, 5, 6, 7, 8])
    # },

    **{
        f'puccinia day {i}': [f'csv/sep-wheat-3/wheat-puccinia/wheat_day{i}_experiment_000']
        for i in [4, 5, 6, 7, 8]
    },

    **{
        f'health day {pi}': [f'csv/sep-wheat-3/wheat-control/wheat_day{hi}_control_000']
        # for hi, pi in zip([4, 5, 4, 5, 4], [4, 5, 6, 7, 8])
        for hi, pi in zip([4, 5, 6, 7, 8], [4, 5, 6, 7, 8])
    },

}

DYNAMIC_WHEAT_NEW_ALGH = {
    **{
        f'puccinia new day {i}': [f'csv/sep-wheat-new-1/wheat-puccinia/wheat-puccinia-{i}']
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    **{
        f'health new day {i}': [f'csv/sep-wheat-new-1/wheat-control/wheat-control-{i}']
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },

    **{
        f'puccinia day {i}': [f'csv/sep-wheat-1/wheat-puccinia/wheat-puccinia-{i}']
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    **{
        f'health day {i}': [f'csv/sep-wheat-1/wheat-control/wheat-control-{i}']
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }
}

DYNAMIC_POTATO_CHECK_FULL = {
    # **{
    #     f'potato health day {day}': [f'csv/control/gala-control-bp-{day}_000']
    #     for day in [1, 2, 3, 4]
    # },
    # **{
    #     f'potato phyto day {day}':
    #         [
    #             *[
    #                 f'csv/phytophthora/gala-phytophthora-bp-{day}_000'
    #             ],
    #             *[
    #                 f'csv/phytophthora/gala-phytophthora-bp-{day + 4}-{day}_000'
    #             ]
    #         ]
    #     for day in [1, 2, 3, 4]
    # },

    **{
        f'potato health day {day}': [f'csv/potato-singles-1/singles-1/gala-control/gala-control-{day}_000']
        for day in [0, 1, 3, 4, 5, 6, 7]
    },
    **{
        f'potato phyto day {day}': [f'csv\potato-singles-1\singles-1\gala-phytophthora\gala-phytophthora-{day}_000']
        for day in [0, 1, 3, 4, 5, 6, 7]
    }

}
