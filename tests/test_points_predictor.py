import pandas

import espn_fantasy_rugby.points_predictor as pp

player_info = pandas.DataFrame({
    'NAME': ['B Kinghorn', 'G North', 'H Slade'],
    'TEAM': ['SCO', 'WAL', 'ENG'],
    'OPP': ['ITA', 'FRA', 'IRE'],
    'POS': ['OB', 'OB', 'C'],
    'MDP': [28, 18, 14],
    'SM': [1, 0, 1]
})

player_info_round = pandas.DataFrame({
    'NAME': ['B Kinghorn', 'G North', 'H Slade'],
    'TEAM': ['SCO', 'WAL', 'ENG'],
    'OPP': ['ITA', 'FRA', 'IRE'],
    'POS': ['OB', 'OB', 'C'],
    'MDP': [28, 18, 14],
    'SM': [1, 0, 1],
    'ROUND': [1, 2, 3]
})


def test_get_starters():

    test_df = pandas.DataFrame({
        'NAME': ['B Kinghorn', 'H Slade'],
        'TEAM': ['SCO', 'ENG'],
        'OPP': ['ITA', 'IRE'],
        'POS': ['OB', 'C'],
        'MDP': [28, 14],
        'SM': [1, 1]
    })

    assert pp.get_starters(player_info).equals(test_df)


def test_get_player_info_previous_rounds():

    test_df = pandas.DataFrame({
        'NAME': ['B Kinghorn', 'G North'],
        'TEAM': ['SCO', 'WAL'],
        'OPP': ['ITA', 'FRA'],
        'POS': ['OB', 'OB'],
        'MDP': [28, 18],
        'SM': [1, 0],
        'ROUND': [1, 2,]
    })

    assert pp.get_player_info_previous_rounds(player_info_round, 3).equals(test_df)
