import json
import os
from pathlib import Path

def dump_data(data, fn, mode='w'):
    if not isinstance(fn, Path):
        fn = Path(fn)
    fp = fn.parent

    if not os.path.exists(fp):
        os.makedirs(fp, exist_ok=True)
    with open(fn, mode) as f:
        f.writelines(data)

def json_write(data, path:Path, indent=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode='w+') as file:
        json.dump(data, file, indent=indent)


def dump_dict_to_json(data:dict, fn, exclude_keys=None, include_keys=None):
    keys = set(data.keys())
    if include_keys is not None:
        if isinstance(include_keys, str):
            include_keys = [include_keys]
        keys = keys.intersection(set(include_keys))
    if exclude_keys is not None:
        if isinstance(exclude_keys, str):
            exclude_keys = [exclude_keys]
        keys = keys.difference(set(exclude_keys))
    _data = {}
    for k in keys:
        _data[k] = data[k]
    json_write(_data, fn)
    
