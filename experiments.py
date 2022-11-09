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

WHEAT_1 = {
    'health1': [
        f'csv/sep-wheat-1/wheat-control/wheat-control-{i}'
        for i in range(4, 10, 1)
    ],
    'puccinia phyto1': [
        f'csv/sep-wheat-1/wheat-puccinia/wheat-puccinia-{i}'
        for i in range(4, 10, 1)
    ]
}

WHEAT_2 = {
    'health2': [
        f'csv/sep-wheat-2/wheat-control/wheat-control-{i}_000'
        for i in range(4, 10, 1)
    ],
    'puccinia phyto2': [
        f'csv/sep-wheat-2/wheat-puccinia/wheat-puccinia-{i}_000'
        for i in range(4, 10, 1)
    ]
}

WHEAT_3 = {
    'health3': [
        f'csv/sep-wheat-3/wheat-control/wheat_day{i}_control_000'
        for i in range(4, 10, 1)
    ],
    'puccinia phyto3': [
        f'csv/sep-wheat-3/wheat-puccinia/wheat_day{i}_experiment_000'
        for i in range(4, 10, 1)
    ]
}
