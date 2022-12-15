POTATO_OLD = {
    'health': [
        'csv/control/gala-control-bp-1_000',
        'csv/control/gala-control-bp-2_000',
        'csv/control/gala-control-bp-3_000',
        'csv/control/gala-control-bp-4_000',
    ],
    'phyto1': [
        'csv/phytophthora/gala-phytophthora-bp-1_000',
        'csv/phytophthora/gala-phytophthora-bp-5-1_000',
        # 'csv/phytophthora-ps-2_2-5_2/gala-phytophthora-2_2-5_2-1_000'
    ],
    'phyto2': [
        'csv/phytophthora/gala-phytophthora-bp-2_000',
        'csv/phytophthora/gala-phytophthora-bp-6-2_000',
        # 'csv/phytophthora-ps-2_2-5_2/gala-phytophthora-2_2-5_2-2_000'
    ]
}

POTATO_NEW = {
    'health new': [
        'csv/potato-singles-1/singles-1/gala-control/gala-control-0_000',
        'csv/potato-singles-1/singles-1/gala-control/gala-control-1_000',
    ],
    'phyto1 new': [
        'csv\potato-singles-1\singles-1\gala-phytophthora\gala-phytophthora-0_000',
        'csv\potato-singles-1\singles-1\gala-phytophthora\gala-phytophthora-5_000',
    ],
    'phyto2 new': [
        'csv\potato-singles-1\singles-1\gala-phytophthora\gala-phytophthora-1_000',
        'csv\potato-singles-1\singles-1\gala-phytophthora\gala-phytophthora-6_000',
    ]
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

DYNAMIC_CHECK = {
    # **{
    #     f'puccinia day {i}': [f'csv/sep-wheat-1/wheat-puccinia/wheat-puccinia-{i}']
    #     for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # },
    # **{
    #     f'health day {i}': [f'csv/sep-wheat-1/wheat-control/wheat-control-{i}']
    #     for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # },

    **{
        f'puccinia day {i}': [f'csv/sep-wheat-2/wheat-puccinia/wheat-puccinia-{i}_000']
        for i in [4, 5, 6, 7, 8]
    },

    **{
        f'health day {pi}': [f'csv/sep-wheat-2/wheat-control/wheat-control-{hi}_000']
        for hi, pi in zip([0, 2, 0, 2, 0], [4, 5, 6, 7, 8])
    },

    # **{
    #     f'puccinia day {i}': [f'csv/sep-wheat-3/wheat-puccinia/wheat_day{i}_experiment_000']
    #     for i in [4, 5, 6, 7, 8]
    # },
    #
    # **{
    #     f'health day {pi}': [f'csv/sep-wheat-3/wheat-control/wheat_day{hi}_control_000']
    #     for hi, pi in zip([4, 5, 4, 5, 4], [4, 5, 6, 7, 8])
    # },

}
