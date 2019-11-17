import pandas

import espn_fantasy_rugby.team_selector as ts


player_info_round = pandas.DataFrame({
    'NAME': ['B Kinghorn', 'G North', 'H Slade', 'J Gray', 'A Bonne', 'R Signor', 'J Connor'],
    'TEAM': ['SCO', 'WAL', 'ENG', 'SCO', 'FRA', 'ITA', 'IRE'],
    'OPP': ['ITA', 'FRA', 'IRE', 'ITA', 'WAL', 'SCO', 'ENG'],
    'POS': ['OB', 'OB', 'C', 'SR', 'FH', 'FR', 'SH'],
    'MDP': [28, 18, 14, 10, 6, 12, 15],
    'ROUND': [2, 2, 2, 2, 2, 2, 2],
    'PREDICTION': [26, 15, 12, 11, 12, 14, 18]
})


def test_add_nationality_constraint():

    test_team = pandas.DataFrame({
        'NAME': ['A Bonne', 'B Kinghorn', 'G North', 'H Slade', 'J Connor', 'R Signor'],
        'TEAM': ['FRA', 'SCO', 'WAL', 'ENG', 'IRE', 'ITA'],
        'POS': ['FH', 'OB', 'OB', 'C', 'SH', 'FR'],
        'MDP': [6, 28, 18, 14, 15, 12]
    })

    prob = ts.create_team_selection_problem(player_info_round)
    prob = ts.add_nationality_constraint(prob=prob, player_forecasts=player_info_round, threshold=1)

    team_selection = ts.get_team(prob=prob, player_forecasts=player_info_round)

    assert test_team.equals(team_selection)


def test_add_position_constraint():

    test_team = pandas.DataFrame({
        'NAME': ['A Bonne', 'B Kinghorn', 'H Slade', 'J Connor', 'J Gray', 'R Signor'],
        'TEAM': ['FRA', 'SCO', 'ENG', 'IRE', 'SCO', 'ITA'],
        'POS': ['FH', 'OB', 'C', 'SH', 'SR', 'FR'],
        'MDP': [6, 28, 14, 15, 10, 12]
    })

    pos_thresholds = {
        'OB': 1,
        'C': 1,
        'FH': 1,
        'SH': 1,
        'BR': 1,
        'SR': 1,
        'FR': 1
    }

    prob = ts.create_team_selection_problem(player_info_round)
    prob = ts.add_position_constraint(prob=prob, player_forecasts=player_info_round, pos_caps=pos_thresholds)

    team_selection = ts.get_team(prob=prob, player_forecasts=player_info_round)

    assert test_team.equals(team_selection)