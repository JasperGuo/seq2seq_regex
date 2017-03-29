# coding=utf8


def get_value(dict_data, key, default=None):
    """
    retrieve value from the dict
    :param dict_data:
    :param key:
    :param default:
    :return:
    """
    if key not in dict_data:
        return default
    return dict_data[key]