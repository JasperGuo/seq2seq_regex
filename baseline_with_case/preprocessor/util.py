# coding=utf8

import json


def read_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_data(file_path, data):
    with open(file_path, "w") as f:
        return f.write(json.dumps(data, indent=4))