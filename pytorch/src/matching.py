import json
import os
from enum import Enum
from random import choice
from classes.argument import ArgType, Argument
from classes.database import TorchDatabase
from classes.argdef import ArgDef
from munkres import Munkres, print_matrix
from classes.torch_api import TorchAPI, TorchArgument
from classes.torch_library import TorchLibrary
from itertools import chain, combinations
from constant.keys import *
from utils.loader import load_data
from utils.printer import dump_data
from os.path import join

root_dir = "../output"

class ResultType(Enum):
    NOT_EQUIVALENT = 1
    SUCCESS = 2
    FAIL = 3
    BUG = 4
    ERROR = 5


PossibleValue = {
    ArgType.INT: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    ArgType.FLOAT: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    ArgType.STR: Argument._str_values,
    ArgType.BOOL: [True, False],
    ArgType.NULL: [None],
}

TorchDatabase.database_config("127.0.0.1", 27017, "torch")

def is_uncovered_api(api):
    covered_api_list = load_data(join(root_dir, "covered_api.txt"), multiline=True)
    return api not in covered_api_list

def count_results(results: list["ResultType"]):
    """
    Count the number of BUG, FAIL and SUCCESS
    Return (#bug, #fail, #success)
    """
    neq_count = 0
    fail_count = 0
    success_count = 0
    err_count = 0
    bug_count = 0
    for result in results:
        neq_count += (result == ResultType.NOT_EQUIVALENT)
        fail_count += (result == ResultType.FAIL)
        success_count += (result == ResultType.SUCCESS)
        err_count += (result == ResultType.ERROR)
        bug_count += (result == ResultType.BUG)
    return {
        "neq": neq_count,
        "fail": fail_count,
        "success": success_count,
        "err": err_count,
        "bug": bug_count,
    }


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def run_and_check(code,
                  equal_type=1,
                  strict_mode=False,
                  neq_dir="output/neq",
                  fail_dir="output/fail",
                  success_dir="output/success",
                  err_dir="output/err",
                  bug_dir="output/bug"):
    """
    @param code: the code will be executed, it should contain results[RES_1, RES_2, ERR_1, ERR_2]
    @param equal_type: if it is 1, compare the value of RES_1 and RES_2; otherwise, only compare the correctness
    """
    def allow_error(error_message):
        errors = [
            "not implement", "invalid combination of argument",
            "a bool tensor is not supported",
            "not intended to support complex numbers", "not supported for"
        ]
        for error in errors:
            if error in error_message:
                return True
        return False

    internal_error = "INTERNAL ASSERT FAILED"
    write_code = "import torch\nresults = dict()\n" + code + "\nprint(results)\n"
    with open("log.py", "w") as f:
        f.write(write_code)

    res_type = ResultType.FAIL
    tgt_state = False
    results, error = TorchLibrary.run_code(code)
    if error:
        TorchLibrary.write_to_dir(err_dir, write_code)
        res_type = ResultType.ERROR
    else:
        tgt_state = (results[ERR_2] == None)
        if results[ERR_1] and results[ERR_2]:
            # check
            if internal_error in results[ERR_1] or internal_error in results[ERR_2]:
                TorchLibrary.write_to_dir(bug_dir, write_code)
                res_type = ResultType.BUG
            else:
                # TorchLibrary.write_to_dir(fail_dir, write_code)
                res_type = ResultType.FAIL
        elif not results[ERR_1] and results[ERR_2]:
            if internal_error in results[ERR_2]:
                TorchLibrary.write_to_dir(bug_dir, write_code)
                res_type = ResultType.BUG
            elif not strict_mode and allow_error(results[ERR_2]):
                # TorchLibrary.write_to_dir(fail_dir, write_code)
                res_type = ResultType.FAIL
            else:
                TorchLibrary.write_to_dir(neq_dir, write_code)
                res_type = ResultType.NOT_EQUIVALENT
        elif results[ERR_1]:
            # TorchLibrary.write_to_dir(fail_dir, write_code)
            res_type = ResultType.FAIL
        elif equal_type == 1:
            if TorchLibrary.is_equal(results[RES_1], results[RES_2], 1e-5):
                # TorchLibrary.write_to_dir(success_dir, write_code)
                res_type = ResultType.SUCCESS
            else:
                TorchLibrary.write_to_dir(neq_dir, write_code)
                res_type = ResultType.NOT_EQUIVALENT
        else:
            # TorchLibrary.write_to_dir(success_dir, write_code)
            res_type = ResultType.SUCCESS
    return res_type, tgt_state


def match_argument(args_A: list['ArgDef'], args_B: list['ArgDef']):
    """
    map argument definition list A to B
    return a mapping list: [(index_A, index_B)]
    """
    sim = ArgDef.similarity(args_A, args_B)
    sim_matrix = [[5 - y for y in x] for x in sim]
    m = Munkres()
    indices = m.compute(sim_matrix)
    indices.sort(key=lambda x: x[1])
    # print(indices)
    return indices


def verify(api_A: "TorchAPI",
           api_B: "TorchAPI",
           num,
           equal_type,
           fuzz,
           indices,
           neq_dir,
           fail_dir,
           success_dir,
           err_dir,
           bug_dir,
           strict_mode=False):
    os.makedirs(neq_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(bug_dir, exist_ok=True)
    res = []
    records = TorchDatabase.get_all_records(api_A.api)

    write_to_db = (not fuzz) and is_uncovered_api(api_B.api)

    for k in range(num):
        if fuzz:
            api_A.get_invocation(choice(records))
            api_A.mutate()
        else:
            if k >= len(records):
                break
            api_A.get_invocation(records[k])
        tgt_record = {}
        code = api_A.to_code(use_try=True,
                             res=f"results[\"{RES_1}\"]",
                             error_res=f"results[\"{ERR_1}\"]")

        arg_str = ""
        input_tensor = ""
        tgt_record_index = 0
        for pair in indices:
            i = pair[1]
            arg_A = api_A.arg_defs[pair[0]]
            temp_name = ""
            temp_arg = arg_A.case

            if temp_arg == None:
                continue
            else:
                if temp_arg.type == ArgType.TORCH_TENSOR:
                    if TorchAPI.is_sparse_tensor(api_B.api, i):
                        temp_name = f"{temp_arg.name}_tensor.clone().to_sparse()"
                    else:
                        temp_name = f"{temp_arg.name}_tensor.clone()"
                else:
                    temp_name = temp_arg.name
            arg_B = api_B.arg_defs[i]
            if arg_B.name == "_input_tensor":
                # check whether the current argument of A is tensor
                if temp_arg.type != ArgType.TORCH_TENSOR:
                    input_tensor = "+++"
                else:
                    input_tensor = temp_name
                tgt_record["_input_tensor"] = temp_arg.to_record()
            elif arg_B.is_optional:
                arg_str += f"{arg_B.name}={temp_name},"
                tgt_record[arg_B.name] = temp_arg.to_record()
            else:
                arg_str += f"{temp_name},"
                tgt_record[f"parameter:{tgt_record_index}"] = temp_arg.to_record()
                tgt_record_index += 1
        if api_B.api.startswith("torch.Tensor."):
            if input_tensor == "+++":
                # not a tensor
                continue
            tensor_op = api_B.api.replace("torch.Tensor.", "")
            call = f"results[\"{RES_2}\"] = {input_tensor}.{tensor_op}({arg_str})\n"
        elif api_B.is_class:
            call = f"results[\"{RES_2}\"] = {api_B.api}({arg_str})(*input_signature)\n"
            tgt_record["input_signature"] = api_A.args["input_signature"].to_record()
        else:
            call = f"results[\"{RES_2}\"] = {api_B.api}({arg_str})\n"

        code += TorchAPI.invocation_code("", f"results[\"{ERR_2}\"]", call,
                                         True, False)
        r, tgt_state = run_and_check(code,
                          equal_type=equal_type,
                          strict_mode=strict_mode,
                          neq_dir=neq_dir,
                          fail_dir=fail_dir,
                          success_dir=success_dir,
                          err_dir=err_dir,
                          bug_dir=bug_dir)
        res.append(r)
        if write_to_db and tgt_state and equal_type:
            TorchDatabase.add_record(api_B.api, tgt_record)
            new_apis = load_data(join(root_dir, "new_api.txt"), multiline=True)
            if api_B.api not in new_apis:
                dump_data(api_B.api + "\n", join(root_dir, "new_api.txt"), "a")
    return res


def match_api(A: str,
              B: str,
              num=50,
              equal_type=1,
              fuzz=True,
              neq_dir="output/neq",
              fail_dir="output/fail",
              success_dir="output/success",
              err_dir="output/err",
              bug_dir="output/bug"):
    """map the arguments of api A to api B's arguments"""
    def verify_wrapper(api_A,
                       api_B,
                       indices,
                       neq_dir,
                       fail_dir,
                       success_dir,
                       err_dir,
                       bug_dir,
                       index=None,
                       value=None):
        for _ in range(10):
            res = verify(api_A,
                         api_B,
                         num,
                         equal_type,
                         fuzz,
                         indices,
                         neq_dir,
                         fail_dir,
                         success_dir,
                         err_dir,
                         bug_dir,
                         strict_mode=True)
            if ResultType.NOT_EQUIVALENT in res:
                return False
            elif ResultType.SUCCESS in res:
                return True
        return False
    os.makedirs(neq_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(bug_dir, exist_ok=True)
    api_A = TorchAPI(A)
    api_B = TorchAPI(B)
    results = []

    if len(api_A.arg_defs) == 0 or len(
            api_B.arg_defs) == 0 or (api_A.is_class != api_B.is_class):
        return None, []

    indices = match_argument(api_A.arg_defs, api_B.arg_defs)

    # check whether there is any unmatched argument in source API
    source_matched_indices = [p[0] for p in indices]
    if len(source_matched_indices) < len(api_A.arg_defs):
        for i in range(len(api_A.arg_defs)):
            if i not in source_matched_indices:
                if api_A.arg_defs[i].is_optional:
                    api_A.arg_defs[i].ignore = True
                else:
                    return None, []
    
    # check whether there is any unmatched argument in target API
    target_matched_indices = [p[1] for p in indices]
    if len(target_matched_indices) < len(api_B.arg_defs):
        for i in range(len(api_B.arg_defs)):
            if i not in target_matched_indices and not api_B.arg_defs[i].is_optional:
                return None, []

    if fuzz:
        results = verify(api_A, api_B, num, equal_type, fuzz, indices, neq_dir, fail_dir, success_dir, err_dir, bug_dir)
    else:
        results = verify_wrapper(api_A, api_B, indices, neq_dir, fail_dir, success_dir, err_dir, bug_dir)
    return results, indices

def match_code(A: str,
               code_hint: str,
               num=50,
               neq_dir="output/neq",
               fail_dir="output/fail",
               success_dir="output/success",
               err_dir="output/err",
               bug_dir="output/bug",):
    """match api A and code hint"""
    os.makedirs(neq_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(bug_dir, exist_ok=True)
    api = TorchAPI(A)
    results = []
    for _ in range(num):
        api.get_invocation()
        api.mutate()
        code = api.to_code(res=f"results[\"{RES_1}\"]",
                           error_res=f"results[\"{ERR_1}\"]",
                           use_try=True)
        code += TorchAPI.invocation_code(
            "", f"results[\"{ERR_2}\"]",
            f"results[\"{RES_2}\"] = {code_hint}\n", True, False)
        result = run_and_check(code, neq_dir=neq_dir, fail_dir=fail_dir, success_dir=success_dir, err_dir=err_dir, bug_dir=bug_dir)
        results.append(result)
    return results