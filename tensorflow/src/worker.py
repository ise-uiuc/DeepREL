from pathlib import Path
from program_synthesizor import tf_matching
from program_synthesizor.tf_matching import EqualType, match_api, count_results, test_api
import sys
from os.path import join
from termcolor import colored
import configparser
from classes.database import TFDatabase
from utils.printer import dump_data
from utils.loader import load_data

def verify(api_1, api_2, equal_type_enum:EqualType, root_dir):
    equal_type = int(equal_type_enum)
    log_dir = root_dir / "logs"
    log_file = join(log_dir, f"ver-log-{equal_type}.txt")
    confirmed_bug_file = join(log_dir, f"ver-confirmedbug-{equal_type}.txt")
    candidate_bug_file = join(log_dir, f"ver-candidatebug-{equal_type}.txt")
    error_file = join(log_dir, f"ver-error-{equal_type}.txt")

    relation_candidate_file = log_dir / f"ver-candidate-apis-{equal_type}.txt"
    relation_confirmed_file = log_dir / f"ver-confirmed-apis-{equal_type}.txt"
    
    dump_data(f"{api_1} {api_2}\n", relation_candidate_file, "a")
    
    output_dir = join(root_dir, f"{api_1}+{api_2}+{equal_type}+ver")
    neq_dir = join(output_dir, "neq")
    fail_dir = join(output_dir, "fail")
    success_dir = join(output_dir, "success")
    err_dir = join(output_dir, "err")
    bug_dir = join(output_dir, "bug")
    results = match_api(api_1,
                        api_2,
                        num_inputs=100,
                        num_verify=10,
                        fuzz=False,
                        equal_type=equal_type_enum,
                        neq_dir=neq_dir,
                        fail_dir=fail_dir,
                        success_dir=success_dir,
                        err_dir=err_dir,
                        bug_dir=bug_dir)
    verified_argmaps = []
    log_fn = join(root_dir, "logs", "log.txt")
    dump_data(f"{api_1} {api_2}\n", log_fn, "a")
    success_flag = False
    if results == [] or results == None:
        dump_data("False\n", log_fn, "a")
    else:
        for flag, r in results:
            dump_data(f"{flag} {str(r)}\n", log_fn, "a")
            if flag:
                verified_argmaps.append(r)
                success_flag = True
    equiv = "value" if equal_type == 1 else "status"
    if success_flag:
        dump_data(f"{api_1} {api_2}\n", relation_confirmed_file, "a")
        print(colored(f"{api_1} {api_2} {equiv}", "green"))
    else:
        print(colored(f"{api_1} {api_2} {equiv}", "red"))
    return verified_argmaps




def test(api_1, api_2, argmap, num, equal_type_enum:EqualType, root_dir):
    equal_type = int(equal_type_enum)
    log_dir = root_dir / "logs"
    log_file = join(log_dir, f"test-log-{equal_type}.txt")
    confirmed_bug_file = join(log_dir, f"test-confirmedbug-{equal_type}.txt")
    candidate_bug_file = join(log_dir, f"test-candidatebug-{equal_type}.txt")
    error_file = join(log_dir, f"test-error-{equal_type}.txt")

    relation_candidate_file = log_dir / f"test-candidate-apis-{equal_type}.txt"
    relation_confirmed_file = log_dir / f"test-confirmed-apis-{equal_type}.txt"

    output_dir = join(root_dir, f"{api_1}+{api_2}+{equal_type}+test")
    neq_dir = join(output_dir, "neq")
    fail_dir = join(output_dir, "fail")
    success_dir = join(output_dir, "success")
    err_dir = join(output_dir, "error")
    bug_dir = join(output_dir, "bug")

    results = test_api(api_1,
                       api_2,
                       argmap,
                       num_inputs=num,
                       equal_type=equal_type,
                       neq_dir=neq_dir,
                       fail_dir=fail_dir,
                       success_dir=success_dir,
                       err_dir=err_dir,
                       bug_dir=bug_dir)
    count = {
        "neq": 0,
        "fail": 0,
        "success": 0,
        "err": 0,
        "bug": 0,
    }
    with open(relation_candidate_file, "a") as f:
        f.write(f"{api_1} {api_2}\n")

    with open(log_file, "a") as f:
        f.write(f"{api_1} {api_2}\n")
        if results == None:
            f.write("REJECT\n")
        else:
            c = count_results(results)
            print(c)
            f.write(str(c) + "\n")
            count["neq"] += c["neq"]
            count["fail"] += c["fail"]
            count["success"] += c["success"]
            count["err"] += c["err"]
            count["bug"] += c["bug"]
    if count["bug"] > 0:
        with open(confirmed_file, "a") as f_confirm:
            f_confirm.write(f"{api_1} {api_2}\n")
    elif count["err"] > 0:
        with open(error_file, "a") as f_err:
            f_err.write(f"{api_1} {api_2}\n")
    elif count["neq"] > 0:
        with open(candidate_bug_file, "a") as f_candidate:
            f_candidate.write(f"{api_1} {api_2}\n")
    elif count["success"] > 0:
        dump_data(f"{api_1} {api_2}\n", relation_confirmed_file, "a")


if __name__ == "__main__":
    api_1 = sys.argv[1]
    api_2 = sys.argv[2]
    num = int(sys.argv[3])
    root_dir = sys.argv[4]
    config_name = sys.argv[5]

    freefuzz_cfg = configparser.ConfigParser()
    freefuzz_cfg.read(f"./config/{config_name}")
    # database configuration
    mongo_cfg = freefuzz_cfg["mongodb"]
    host = mongo_cfg["host"]
    port = int(mongo_cfg["port"])
    TFDatabase.database_config(host, port, mongo_cfg["tf_database"])

    root_dir = Path(root_dir)
    tf_matching.root_dir = root_dir
    from classes.library_def import tf_lib_def
    tf_lib_def.load_apis(lazy=True)

    # Verified arg maps
    v = verify(api_1, api_2, EqualType.VALUE, root_dir)
    if len(v) > 0:
        # match_value
        for argmap in v:
            test(api_1, api_2, argmap, num, EqualType.VALUE, root_dir)
    else:
        v = verify(api_1, api_2, EqualType.STATUS, root_dir)
        if len(v) > 0:
            for argmap in v:
                # match_status
                test(api_1, api_2, argmap, num, EqualType.STATUS, root_dir)
