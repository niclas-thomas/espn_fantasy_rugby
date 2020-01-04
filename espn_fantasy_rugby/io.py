import pandas
import unicodedata


def clean_names(name):
    """

    :param name:
    :return:
    """
    clean_name = ''.join(
        char for char in name if unicodedata.category(char)[0] != 'P'
    )

    return clean_name


def read_espn_data(filename):
    """

    :param filename:
    :return:
    """

    espn_data = pandas.concat(
        pandas.read_excel(
            filename,
            sheet_name=None
        ),
        ignore_index=True
    )

    espn_data['NAME'] = espn_data['NAME'].apply(lambda x: clean_names(x))

    return espn_data
