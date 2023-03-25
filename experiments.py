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
    #     for hi, pi in zip([0, 2, 0, 2, 0], [4, 5, 6, 7, 8])
    #     # for hi, pi in zip([0, 2, 3, 4, 5], [4, 5, 6, 7, 8])
    # },

    **{
        f'puccinia day {i}': [f'csv/sep-wheat-3/wheat-puccinia/wheat_day{i}_experiment_000']
        for i in [4, 5, 6, 7, 8]
    },

    **{
        f'health day {pi}': [f'csv/sep-wheat-3/wheat-control/wheat_day{hi}_control_000']
        for hi, pi in zip([4, 5, 4, 5, 4], [4, 5, 6, 7, 8])
        # for hi, pi in zip([4, 5, 6, 7, 8], [4, 5, 6, 7, 8])
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
