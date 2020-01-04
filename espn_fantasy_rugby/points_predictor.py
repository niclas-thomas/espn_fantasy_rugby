import pandas
import sklearn.model_selection
import sklearn.ensemble


def clean_names(data):
    """

    :param data:
    :return:
    """

    data['NAME'] = data['NAME'].replace('-', ' ')
    data['NAME'] = data['NAME'].replace('\'', ' ')
    data['NAME'] = data['NAME'].replace('.', '')

    return data


def get_starters(data):
    """

    :param data:
    :return:
    """

    # data = data[data['SM'] == 1].reset_index(drop=True)

    return data


def get_features_and_target(data, features, target):
    """

    :param data:
    :param features:
    :param target:
    :return:
    """

    data = data[features+[target]]

    return data


def get_player_info_previous_rounds(data, tournament_round):
    """

    :param data:
    :param tournament_round:
    :return:
    """

    data = data[data['ROUND'] < tournament_round].reset_index(drop=True)

    return data


def get_player_info_current_round(data, tournament_round):
    """

    :param data:
    :param tournament_round:
    :return:
    """

    data = data[data['ROUND'] == tournament_round].reset_index(drop=True)

    return data


def generate_dummy_variables(data, features_to_encode):
    """

    :param data:
    :param features_to_encode:
    :return:
    """

    dummy_data = pandas.get_dummies(
        data,
        columns=features_to_encode
    )

    return dummy_data


def get_model(trainingdata, target_variable):
    """

    :param trainingdata:
    :param target_variable:
    :return:
    """
    model = sklearn.ensemble.RandomForestRegressor(random_state=1)
    parameters = {
        'n_estimators': [5, 15]
    }
    grid = sklearn.model_selection.GridSearchCV(
        model,
        parameters,
        iid=True,
        cv=3
    )

    x_train = trainingdata[[col for col in trainingdata.columns if col != target_variable]]
    y_train = trainingdata[target_variable]

    grid.fit(x_train, y_train)

    return grid.best_estimator_


def get_player_points_predictions(trainedmodel, data, target):
    """

    :param trainedmodel:
    :param x_test:
    :return:
    """

    features = [col for col in data.columns if col != target]
    data = data[features]

    exact_predictions = trainedmodel.predict(data)
    # hack to include all variables inc 0s
    rounded_predictions = [round(i+1) for i in exact_predictions]

    return rounded_predictions


def get_results(espn_data, features, target, stage):

    espn_data = clean_names(espn_data)
    espn_data = get_starters(espn_data)

    training_data = get_player_info_previous_rounds(espn_data, stage)
    training_data = get_features_and_target(training_data, features, target)
    training_data = generate_dummy_variables(training_data, features)
    model = get_model(training_data, target)

    explanatory_data = get_player_info_current_round(espn_data, stage)
    current_round_data = get_features_and_target(explanatory_data, features, target)
    current_round_data = generate_dummy_variables(current_round_data, features)
    predictions = get_player_points_predictions(model, current_round_data, target)
    explanatory_data['PREDICTION'] = predictions

    return explanatory_data
