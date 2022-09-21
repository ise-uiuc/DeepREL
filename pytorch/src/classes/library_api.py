import os
from classes.database import TorchDatabase
from constant.paths import torch_api_dir
TorchDatabase.database_config("localhost", 27017, "test")

class Library:
    pass

class LibraryAPI:
    pass

class TFLibrary:
    pass

api_list = []
for module in os.listdir(torch_api_dir):
    with open(torch_api_dir / module) as f:
        for api in f.read().split("\n"):
            if api.startswith("torch.") and api != "torch.Tensor":
                api_list.append(api)

class TorchLibrary:
    func_api_names = api_list
    std_api_names = api_list # FIXME:
    pass
