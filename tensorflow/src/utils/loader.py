import inspect
import json
from pathlib import Path
from constant.paths import all_tf_symbol_file

def load_data(fn, multiline=False):
    try:
        with open(fn, 'r') as f:
            if multiline:
                data = f.readlines()
            else:
                data = f.read()
    except Exception as e:
        return None
    return data

def load_api_names(library_name='tf', file_name=all_tf_symbol_file):
    import tensorflow as tf
    if library_name == 'tf':
        apis = load_data(file_name, multiline=True)
        apis = [api.strip() for api in apis]
        return apis
    else:
        raise ValueError(f"{library_name} not supported")

def load_apis(library_name):
    pass


def json_load(path: Path):
    with path.open() as file:
        return json.load(file)
