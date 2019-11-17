import pulp
import pandas

TEAM_COLORS = {
    'WAL':'red',
    'ENG':'white',
    'IRE':'green',
    'FRA':'mediumblue',
    'SCO':'navy',
    'ITA':'dodgerblue'
}

TEAM_FONT_COLORS = {
    'WAL':'white',
    'ENG':'black',
    'IRE':'white',
    'FRA':'white',
    'SCO':'white',
    'ITA':'white'
}


def create_pulp_varnames(x):

    return 'Player_' + x.replace(" ", "_")


def create_team_selection_problem(player_forecasts):
    """

    :param player_forecasts:
    :return:
    """

    player_forecasts['VARNAME'] = player_forecasts['NAME'].apply(lambda x: create_pulp_varnames(x))

    player_forecasts = player_forecasts.sort_values('NAME')
    players = player_forecasts['NAME']
    player_points = player_forecasts[['NAME', 'PREDICTION']].set_index('NAME').to_dict()['PREDICTION']
    player_vars = pulp.LpVariable.dicts(
        name="Player",
        indexs=players,
        lowBound=0,
        upBound=1,
        cat='Integer'
    )

    prob = pulp.LpProblem("ESPN Fantasy Rugby", pulp.LpMaximize)
    prob += pulp.lpSum([player_points[i] * player_vars[i] for i in players]), "Total Points Scored"

    return prob


def add_nationality_constraint(prob, player_forecasts, threshold):

    wal_nation = {
        name: 1 if team == 'WAL' else 0 for name, team in zip(player_forecasts['VARNAME'], player_forecasts['TEAM'])
    }
    eng_nation = {
        name: 1 if team == 'ENG' else 0 for name, team in zip(player_forecasts['VARNAME'], player_forecasts['TEAM'])
    }
    ire_nation = {
        name: 1 if team == 'IRE' else 0 for name, team in zip(player_forecasts['VARNAME'], player_forecasts['TEAM'])
    }
    fra_nation = {
        name: 1 if team == 'FRA' else 0 for name, team in zip(player_forecasts['VARNAME'], player_forecasts['TEAM'])
    }
    sco_nation = {
        name: 1 if team == 'SCO' else 0 for name, team in zip(player_forecasts['VARNAME'], player_forecasts['TEAM'])
    }
    ita_nation = {
        name: 1 if team == 'ITA' else 0 for name, team in zip(player_forecasts['VARNAME'], player_forecasts['TEAM'])
    }

    prob += pulp.lpSum(
        [wal_nation[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= threshold, "WAL_Requirement"
    prob += pulp.lpSum(
        [eng_nation[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= threshold, "ENG_Requirement"
    prob += pulp.lpSum(
        [ire_nation[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= threshold, "IRE_Requirement"
    prob += pulp.lpSum(
        [fra_nation[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= threshold, "FRA_Requirement"
    prob += pulp.lpSum(
        [sco_nation[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= threshold, "SCO_Requirement"
    prob += pulp.lpSum(
        [ita_nation[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= threshold, "ITA_Requirement"

    return prob


def add_position_constraint(prob, player_forecasts, pos_caps):

    ob_position = {
        name: 1 if pos == 'OB' else 0 for name, pos in zip(player_forecasts['VARNAME'], player_forecasts['POS'])
    }
    c_position = {
        name: 1 if pos == 'C' else 0 for name, pos in zip(player_forecasts['VARNAME'], player_forecasts['POS'])
    }
    fh_position = {
        name: 1 if pos == 'FH' else 0 for name, pos in zip(player_forecasts['VARNAME'], player_forecasts['POS'])
    }
    sh_position = {
        name: 1 if pos == 'SH' else 0 for name, pos in zip(player_forecasts['VARNAME'], player_forecasts['POS'])
    }
    br_position = {
        name: 1 if pos == 'BR' else 0 for name, pos in zip(player_forecasts['VARNAME'], player_forecasts['POS'])
    }
    sr_position = {
        name: 1 if pos == 'SR' else 0 for name, pos in zip(player_forecasts['VARNAME'], player_forecasts['POS'])
    }
    fr_position = {
        name: 1 if pos == 'FR' else 0 for name, pos in zip(player_forecasts['VARNAME'], player_forecasts['POS'])
    }

    prob += pulp.lpSum(
        [ob_position[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= pos_caps['OB'], "OB"
    prob += pulp.lpSum(
        [c_position[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= pos_caps['C'], "C"
    prob += pulp.lpSum(
        [fh_position[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= pos_caps['FH'], "FH"
    prob += pulp.lpSum(
        [sh_position[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= pos_caps['SH'], "SH"
    prob += pulp.lpSum(
        [br_position[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= pos_caps['BR'], "BR"
    prob += pulp.lpSum(
        [sr_position[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= pos_caps['SR'], "SR"
    prob += pulp.lpSum(
        [fr_position[i] * prob.variablesDict()[i] for i in prob.variablesDict().keys()]) <= pos_caps['FR'], "FR"

    return prob


def get_team(prob, player_forecasts):

    player_forecasts = player_forecasts.sort_values('NAME')
    players = player_forecasts['NAME']
    player_positions = player_forecasts[['NAME', 'POS']].set_index('NAME').to_dict()['POS']
    player_nationality = player_forecasts[['NAME', 'TEAM']].set_index('NAME').to_dict()['TEAM']
    player_variables = {str(value): key for key, value in zip(players, prob.variables())}
    prob.solve()

    team = pandas.DataFrame()

    for v in prob.variables():
        if v.varValue == 1:
            name = player_variables[v.name]
            team = team.append(pandas.DataFrame({
                'NAME': [name],
                'TEAM': [player_nationality[name]],
                'POS': [player_positions[name]]
            }))
    team = team.merge(player_forecasts[['NAME', 'MDP']], on='NAME')

    return team


def get_predicted_team_points(prob):

    return pulp.value(prob.objective)


def get_actual_team_points(team):

    return team['MDP'].sum()


def show_pitch(ax):
    # Pitch Markings
    ax.axhline(4, linestyle='-', color='white')
    ax.axhline(1.5, linestyle='-', color='white')
    ax.axhline(6.5, linestyle='-', color='white')
    ax.axhline(3.2, linestyle='--', color='white')
    ax.axhline(4.8, linestyle='--', color='white')
    ax.axvline(0.2, linestyle='--', color='white')
    ax.axvline(3.8, linestyle='--', color='white')
    ax.axvline(0.75, linestyle='-.', color='white')
    ax.axvline(3.25, linestyle='-.', color='white')

    # Pitch Color
    ax.set_facecolor('limegreen')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def show_team(team, ax):
    show_pitch(ax)

    team = team.sort_values('POS').reset_index(drop=True)

    coords = [
        (1, 5.5),
        (2, 5),
        (3, 5.5),
        (1.5, 2),
        (2.5, 2),
        (2, 3),
        (1, 7),
        (2, 7),
        (3, 7),
        (1, 1),
        (2, 0.5),
        (3, 1),
        (2, 4),
        (1.5, 6.25),
        (2.5, 6.25)
    ]

    for ix, row in team.iterrows():
        ax.text(
            coords[ix][0],
            coords[ix][1],
            '{} ({})'.format(row['NAME'], row['MDP']),
            ha='center',
            va='center',
            fontsize=11,
            color=TEAM_FONT_COLORS[row['TEAM']],
            bbox=dict(
                boxstyle='round,pad=1',
                facecolor=TEAM_COLORS[row['TEAM']],
                alpha=1
            )
        )

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 8)

    return ax