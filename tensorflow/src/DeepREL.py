from pathlib import Path
from classes.database import TFDatabase
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from os.path import join
import json
import subprocess
# import torch
import inspect
import time
from termcolor import colored
from program_synthesizor.tf_matching import EqualType
from classes.tf_api import TFAPI
from random import choice

import configparser

from utils.loader import load_data
from utils.printer import dump_data

def timestamp(info, root_dir):
    dump_data(str(time.time()) + " " + info + "\n", join(root_dir, "logs", "time.txt"), "a")

import sys
from constant.paths import expr_dir, tf_api_match_dir

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
    TFDatabase.database_config(host, port, mongo_cfg["tf_database"])
    from classes.library_def import tf_lib_def
    tf_lib_def.load_apis(lazy=True)
    # DeepREL
    DeepREL_cfg = freefuzz_cfg["DeepREL"]
    test_number = int(DeepREL_cfg["test_number"])
    top_k = int(DeepREL_cfg["top_k"])
    iteration = int(DeepREL_cfg["iteration"])

    tf_match_dir = tf_api_match_dir
    mdir = tf_api_match_dir 
    
    db = TFDatabase
    max_iter = iteration
    debug = False

    for i in range(0, max_iter):
        root_dir = expr_dir / f"output-{i}"
        os.makedirs(root_dir, exist_ok=True)

        from program_synthesizor.tf_matching import root_dir as local_root_dir
        local_root_dir = root_dir

        log_dir = root_dir / "logs"

        # Preprocess: Get api list
        covered_api_list = db.get_api_list()
         
        covered_file = join(log_dir, "covered_api.txt")
        if not os.path.exists(covered_file):
            dump_data("\n".join(covered_api_list), join(log_dir, "covered_api.txt"))

        covered_api_list = load_data(covered_file, multiline=True)
        covered_api_list = [x.strip() for x in covered_api_list]
        
        new_api_file = join(log_dir, "new_api.txt")
        if not os.path.exists(new_api_file):
            dump_data("", new_api_file)

        if i == 0:
            api_list = covered_api_list
        else:
            api_list = load_data(join(expr_dir / f"output-{i-1}" / "logs", "new_api.txt"), multiline=True)
            api_list = [a.strip() for a in api_list]
        
            
        api_tot = len(api_list)
        print(f"Iteration {i} : {api_tot} apis")
        if not os.path.exists(join(root_dir, "logs", "time.txt")):
            timestamp(f"init covered API={len(api_list)}", root_dir)
        api_list.sort()

        for i, api_name in enumerate(api_list):
            if (i % 100) == 0:
                print(f"[{i} / {api_tot}]")
            pred = []
            api_js = os.path.join(mdir, api_name + '.json')
            if os.path.exists(api_js):
                data = load_data(api_js)
                data = data.split("\n")
                for line in data:
                    line = line.strip()
                    if line == "": continue
                    tgt_api_name, score = json.loads(line.strip())
                    pred.append(tgt_api_name)
            
            count = 0
            
            for similar_api_name in pred:
                if count == top_k: 
                    break
                    
                id = f"{api_name}+{similar_api_name}"
                if similar_api_name == api_name: continue
                count += 1

                equal_type = int(EqualType.VALUE)
                pair_output_dir = os.path.join(root_dir, f"{api_name}+{similar_api_name}+{equal_type}+ver")
                if os.path.isdir(pair_output_dir):
                    continue
                try:
                    
                    res = subprocess.run(["python", "worker.py", api_name, similar_api_name, str(test_number), root_dir, config_name], timeout=300, shell=False)
                except subprocess.TimeoutExpired:
                    with open(join(log_dir, "run-timeout.txt"), "a") as f:
                        f.write(f"{api_name} {similar_api_name}\n")
                except Exception as e:
                    print(colored(e, "red"))
                    with open(join(log_dir, "run-crash.txt"), "a") as f:
                        errmsg = str(e).replace('\n', '')
                        f.write(f"{api_name} {similar_api_name} {errmsg}\n")
                else:
                    if res.returncode != 0:
                        with open(join(log_dir, "run-error.txt"), "a") as f:
                            f.write(f"{api_name} {similar_api_name} {res.returncode}\n")
                    else:
                        pass
                    
                timestamp(f"{api_name} {similar_api_name}", root_dir)
