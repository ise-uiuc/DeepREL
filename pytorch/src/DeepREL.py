from classes.database import TorchDatabase
from classes.torch_api import TorchAPI
import os
from os.path import join
import json
import subprocess
import torch
import inspect
import time
from utils.loader import load_data
from utils.printer import dump_data
import sys
import configparser

if __name__ == "__main__":
    if len(sys.argv) < 2:
        config_name = "demo.conf"
    else:
        config_name = sys.argv[1]
    freefuzz_cfg = configparser.ConfigParser()
    freefuzz_cfg.read(f"./config/{config_name}")

    # database configuration
    mongo_cfg = freefuzz_cfg["mongodb"]
    host = mongo_cfg["host"]
    port = int(mongo_cfg["port"])
    TorchDatabase.database_config(host, port, mongo_cfg["torch_database"])
    # DeepREL
    DeepREL_cfg = freefuzz_cfg["DeepREL"]
    test_number = int(DeepREL_cfg["test_number"])
    top_k = int(DeepREL_cfg["top_k"])
    iteration = int(DeepREL_cfg["iteration"])

    def timestamp(root_dir):
        dump_data(str(time.time()) + "\n", join(root_dir, "time.txt"), "a")

    def is_class(api_name):
        try:
            return inspect.isclass(eval(api_name))
        except Exception:
            return False

    similar_dir = "../torch-candidate"
    for i in range(iteration):
        root_dir = f"../expr/output-{i}"
        os.makedirs(root_dir, exist_ok=True)

        # preprocess
        covered_api_list = TorchDatabase.get_api_list()
        dump_data("\n".join(covered_api_list), join(root_dir, "covered_api.txt"))
        dump_data("", join(root_dir, "new_api.txt"))

        print(f"Iteration: {i} / {iteration}, PyTorch API Num: {len(covered_api_list)}")
        if i == 0:
            api_list = covered_api_list
        else:
            api_list = load_data(join(f"../output-{i-1}", "new_api.txt"), multiline=True)

        # relation test
        timestamp(root_dir)
        for file_name in os.listdir(similar_dir):
            api_name = file_name.replace(".json", "")
            if api_name not in api_list:
                continue
            similar_pairs = []
            with open(join(similar_dir, file_name)) as f:
                for line in f.read().split("\n"):
                    if len(line):
                        similar_pairs.append(json.loads(line))
            # similar_pairs = similar_pairs[:10]
            count = 0
            for pair in similar_pairs:
                if count == top_k:
                    break
                similar_api_name = pair[0]
                similarity = pair[1]
                # if similar_api_name == api_name or similar_api_name not in api_list:
                if similar_api_name == api_name:
                    continue
                count += 1
                # print(api_name, similar_api_name, similarity)

                api_pair_dir = join(root_dir, "{api_name}+{similar_api_name}+rel")
                if os.path.isdir(api_pair_dir):
                    continue
                try:
                    res = subprocess.run(["python", "worker.py", api_name, similar_api_name, str(test_number), root_dir], timeout=300, shell=False)
                except subprocess.TimeoutExpired:
                    dump_data(f"{api_name} {similar_api_name}\n", join(root_dir, "test-run-timeout.txt"), "a")
                except Exception as e:
                    dump_data(f"{api_name} {similar_api_name}\n", join(root_dir, "test-run-crash.txt"), "a")
                else:
                    if res.returncode != 0:
                        dump_data(f"{api_name} {similar_api_name}\n", join(root_dir, "test-run-error.txt"), "a")
                timestamp(root_dir)
