import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from os.path import join
from pprint import pprint
from random import choice, randint
from enum import IntEnum
from pathlib import Path
from typing import List, Tuple, Any
from unittest import result
from classes.argument import ArgType, Argument
from classes.database import Database, TFDatabase
from classes.argdef import ArgDef
from munkres import Munkres, print_matrix
from classes.tf_api import TFAPI, TFArgument
from classes.tf_library import TFLibrary
from classes.library_def import tf_lib_def
from classes.api import API
from classes.argument import Argument
from itertools import chain, combinations
from constant.keys import *
import re
import tensorflow as tf

from utils.loader import load_data
from utils.printer import dump_data
from utils.utils import str_to_value
from utils.probability import do_select_from_db

root_dir = "../output"

class ResultType(IntEnum):
    NOT_EQUIVALENT = 1
    SUCCESS = 2
    FAIL = 3
    BUG = 4
    ERROR = 5

class EqualType(IntEnum):
    VALUE = 1
    STATUS = 2

PossibleValue = {
    ArgType.INT: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    ArgType.FLOAT: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    ArgType.STR: Argument._str_values,
    ArgType.BOOL: [True, False],
    ArgType.NULL: [None],
}

class ArgumentMapping:
    """ 
    Base class for argument matching.

    Args:
        _indices_mapping: Dict[int, int], mapping from tgt indices to src indices
        _src_fixed_value_mapping: Dict[str, Any] mapping from src argname to fixed value
    """
    
    def __init__(self, a:API, b:API, _idmap, _valmap={}):
        self._src_API = a
        self._tgt_API = b
        
        self._indices_mapping = _idmap
        self._src_fixed_value_mapping = _valmap

    def map_args(self, a:API, b:API) -> str:
        """ Map args of a to b, returns code for tensor copying. """
        raise NotImplementedError

    def __str__(self):
        m1 = sorted(list(self._indices_mapping.items()))
        m2 = sorted(list(self._src_fixed_value_mapping.items()))
        m = [m1, m2]
        return json.dumps(m)

    
    def load_from_json(self, s):
        self._indices_mapping = dict()
        self._src_fixed_value_mapping = dict()
        m1, m2 = json.loads(s)
        for i, j in m1:
            self._indices_mapping[i] = j

        
        

class TFArgumentMapping(ArgumentMapping):

    @staticmethod
    def check_unmapped_index(a:TFAPI, b:TFAPI, indices_mapping) -> List['TFArgumentMapping']:
        """ Adds fixed value mapping for source API if there is one unmapped argument.

        Returns a list of possible Mappings, or an empty list if no solution. """

        # Get the mapped indices
        mapped_tgt_index = []
        mapped_src_index = []
        for i, j in indices_mapping.items():
            mapped_tgt_index.append(i)
            mapped_src_index.append(j)
        
        # Check that every target argument (if non optional) is matched
        for i, arg in enumerate(b.arg_defs):
            if i not in mapped_tgt_index:
                if not arg.is_optional:
                    return []
        
        # Check that every source argument (if non optional) is matched
        unmapped_src_index = []
        for j, arg in enumerate(a.arg_defs):
            if j not in mapped_src_index:
                if not arg.is_optional:
                    return []
        return [TFArgumentMapping(a, b, indices_mapping, {})]

    def delete_unmapped_args(self, a:TFAPI, b:TFAPI):
        """ Remove unmapped option args"""
        # Get the mapped indices
        mapped_tgt_index = []
        mapped_src_index = []
        for i, j in self._indices_mapping.items():
            mapped_tgt_index.append(i)
            mapped_src_index.append(j)

        keys = [k for k in a.args.keys()]
        for key in keys:
            arg = a.args[key]
            argdef = a.find_arg_with_name(key)
            argind = a.api_def.find_arg(key)
            # argdef is None if key is input
            if (argdef != None) and argdef.is_optional and (argind not in mapped_src_index):
                a.args.pop(key)

        keys = [k for k in b.args.keys()]
        for key in keys:
            argdef = b.find_arg_with_name(key)
            argind = b.api_def.find_arg(key)
            if (argdef != None) and argdef.is_optional and (argind not in mapped_tgt_index):
                b.args.pop(key)

    def map_args(self, a:TFAPI, b:TFAPI):

        code = ""
        
        ready_args = []
        for i, j in self._indices_mapping.items():
            ready_args.append(i)
            src_arg = a.get_arg(j)
            b.set_arg(i, src_arg)
            barg = b.get_arg(i)
            if barg.is_tensorlike():
                src_tensor_name = barg.var_name
                assert len(barg.var_name) > 0
                barg.var_name += "_cp"
                code += f"{barg.var_name} = tf.identity({src_tensor_name})\n"

        # Handle the unmapped (optional) arguments of API b.
        for i, arg_def in enumerate(b.arg_defs):
            if i in ready_args: continue
            arg = b.get_arg_by_names(arg_def.name, "parameter:"+str(i))
            if arg == None: continue
            assert len(arg.var_name) > 0
            code += arg.to_code(arg.var_name)

        if API_INVOCATION_INPUT_KEY in a.args:
            inputs = a.args[API_INVOCATION_INPUT_KEY]
            b.args[API_INVOCATION_INPUT_KEY] = inputs

        return code




def is_uncovered_api(api):
    covered_api_list = load_data(join(root_dir, "logs", "covered_api.txt"), multiline=True)
    covered_api_list = [a.strip() for a in covered_api_list]
    return api.strip() not in covered_api_list


def count_results(results: List["ResultType"]):
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

def pretty_print(res):
    import numpy as np
    pres = dict()
    for k, v in res.items():
        if tf.is_tensor(v):
            pres[k] = f"{v.shape}, {v.dtype}, {np.average(v.numpy())}"
        else:
            pres[k] = v
    print(pres)
    return pres

pretty_print_str = """def pretty_print(res):
    pres = dict()
    for k, v in res.items():
        if tf.is_tensor(v):
            pres[k] = f"{v.shape}, {v.dtype}, {np.average(v.numpy())}"
        else:
            pres[k] = v
    print(pres)
"""

def run_and_check(code, equal_type=EqualType.VALUE, 
                  strict_mode=False,
                  neq_dir="output/neq",
                  fail_dir="output/fail",
                  success_dir="output/success",
                  err_dir="output/err",
                  bug_dir="output/bug", maxcnt=None,
    verbose=False, record=None, db:Database=None, api_1:str=None, api_2:str=None) -> ResultType:
    """
    @param code: the code will be executed, it should contain results[RES_1, RES_2, ERR_1, ERR_2]
    @param equal_type: if it is 1, compare the value of RES_1 and RES_2; otherwise, only compare the correctness
    """
    def allow_error(error_message):
        errors = [ ] 
        for error in errors:
            if error in error_message:
                return True
        return False

    internal_error = "INTERNAL ASSERT FAILED"
    random_seed = randint(7, 100007)
    code = f"tf.random.set_seed({random_seed})\n" + code

    write_code = "import tensorflow as tf\nimport numpy as np\nresults = dict()\n" + code + \
        pretty_print_str + "\npretty_print(results)\n"
    pair_dir = Path(bug_dir).parent
    with open(pair_dir / "log.py", "w") as f:
        f.write(write_code)
    # print(code)
    
    res_type = ResultType.FAIL
    tgt_state = False
    results, error = tf_lib_def.run_code(code)

    if (error != None) and (results[ERR_1] != None):
        tf_lib_def.write_to_dir(fail_dir, write_code)
        return ResultType.FAIL, tgt_state

            
    if error:
        tf_lib_def.write_to_dir(err_dir, write_code, maxcnt=maxcnt)
        res_type = ResultType.ERROR
    else:
        tgt_state = (results[ERR_2] == None)
        if (results[ERR_1] != None) and (results[ERR_2] != None):
            # check
            if (internal_error != None) and (internal_error in results[ERR_1] or internal_error in results[ERR_2]):
                tf_lib_def.write_to_dir(bug_dir, write_code, maxcnt=maxcnt)
                res_type = ResultType.BUG
            else:
                # tf_lib_def.write_to_dir(fail_dir, write_code)
                res_type = ResultType.FAIL
        elif (results[ERR_1] == None) and (results[ERR_2] != None):
            if internal_error in results[ERR_2]:
                tf_lib_def.write_to_dir(bug_dir, write_code, maxcnt=maxcnt)
                res_type = ResultType.BUG
            elif allow_error(results[ERR_2]):
                # tf_lib_def.write_to_dir(fail_dir, write_code)
                res_type = ResultType.FAIL
            else:
                tf_lib_def.write_to_dir(neq_dir, write_code, maxcnt=maxcnt)
                res_type = ResultType.NOT_EQUIVALENT
        elif results[ERR_1] != None:
            # tf_lib_def.write_to_dir(fail_dir, write_code)
            res_type = ResultType.FAIL
        elif equal_type == EqualType.VALUE:
            if TFLibrary.is_equal(results[RES_1], results[RES_2]):
                # tf_lib_def.write_to_dir(success_dir, write_code)
                res_type = ResultType.SUCCESS
            else:
                tf_lib_def.write_to_dir(neq_dir, write_code, maxcnt=maxcnt)
                res_type = ResultType.NOT_EQUIVALENT
        elif equal_type == EqualType.STATUS:
            # tf_lib_def.write_to_dir(success_dir, write_code)
            res_type = ResultType.SUCCESS
        else:
            raise Exception("[ERROR] Wrong equal_type: " + str(equal_type))
    return res_type, tgt_state


def match_argument(args_A: List['ArgDef'], args_B: List['ArgDef'], verbose=True) -> List[Tuple[int, int]]:
    """
    map argument definition list A to B
    return a mapping list: [(index_A, index_B)]
    """
    sim = ArgDef.similarity(args_A, args_B, verbose=False)
    sim_matrix = [[5 - y for y in x] for x in sim]
    m = Munkres()
    indices = m.compute(sim_matrix)
    indices.sort(key=lambda x: x[1])
    filter_ind = []
    for i, j in indices:
        if args_A[i].name == "**kwargs" or args_B[j].name == "**kwargs": continue
        if args_A[i].name == "*args" or args_B[j].name == "*args": continue
        filter_ind.append((i,j))
    return filter_ind

def verify(api_A: "TFAPI", api_B: "TFAPI", argmap: ArgumentMapping, num, equal_type, fuzz, 
           neq_dir,
           fail_dir,
           success_dir,
           err_dir,
           bug_dir,
           maxcnt=None,
           strict_mode=False,
           verbose=False) -> List[ResultType]:
    os.makedirs(neq_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(bug_dir, exist_ok=True)

    if not isinstance(equal_type, EqualType):
        equal_type = EqualType(equal_type)
    
    write_to_db = (not fuzz) and is_uncovered_api(api_B.api)

    res = []
    records = TFDatabase.get_all_records(api_A.api)
    if len(records) == 0:
        print("[WARNING] no record for {api_A.api}")
    # Reload.
    api_A = TFAPI(api_A.api)
    api_B = TFAPI(api_B.api)
    for k in range(num):
        if fuzz:
            record = choice(records)
            api_A.get_invocation(record)
            api_A.mutate()
        else:
            if k >= len(records):
                break
            api_A.get_invocation(records[k])
        if False:
            print("Load record to API A: record = ")
            pprint(api_A.to_record()) 

        argmap.delete_unmapped_args(api_A, api_B)
        # API A invocation.
        code = api_A.to_code(use_try=True,
                            res=f"results[\"{RES_1}\"]",
                            err_name=ERR_1)

        # Prepare API B argument, copy tensors if necessary.
        code += argmap.map_args(api_A, api_B)
                
        # API B invocation.
        code += api_B.to_code(use_try=True,
                              res=f"results[\"{RES_2}\"]",
                              skip_arg_to_code=True,
                              err_name=ERR_2)
        record_A = api_A.to_record()
        record_B = api_B.to_record()
        r, tgt_state = run_and_check(code,
                          equal_type=equal_type,
                          strict_mode=strict_mode,
                          verbose=verbose,
                          neq_dir=neq_dir,
                          fail_dir=fail_dir,
                          success_dir=success_dir,
                          err_dir=err_dir,
                          bug_dir=bug_dir, maxcnt=maxcnt)
        res.append(r)
        if write_to_db and tgt_state:
            TFDatabase.add_record(api_B.api, record_B)
            new_fn = root_dir / "logs" / "new_api.txt"
            new_apis = load_data(new_fn, multiline=True)
            if new_apis == None: 
                new_apis = []
            else:
                new_apis = [x.strip() for x in new_apis]
                if api_B.api not in new_apis:
                    dump_data(api_B.api + "\n", new_fn, "a")
    if verbose:
        print("verify res: ", res)
    return res

def find_src_ind_from_arg_matching(indices, tid):
    for src_ind, tgt_ind in indices:
        if tgt_ind == tid:
            return src_ind
    return -1

def get_api_map(A: str, B: str):
    api_A = TFAPI(A)
    api_B = TFAPI(B)
    if api_A.api_def == None:
        return None
    if api_B.api_def == None:
        return None
    if len(api_A.arg_defs) == 0 or len(api_B.arg_defs) == 0 or (api_A.is_class != api_B.is_class):
        return None
    
    indices = match_argument(api_A.arg_defs, api_B.arg_defs, verbose=True)
    indices_map = {}
    for i, j in indices: 
        indices_map[j] = i
    
    
    # Try to add fixed value to the mapping
    argmaps = TFArgumentMapping.check_unmapped_index(api_A, api_B, indices_map)
    return argmaps

def match_api(A: str,
              B: str,
              num_inputs=50,
              num_verify=10,
              equal_type=1,
              fuzz=True,
              neq_dir="output/neq",
              fail_dir="output/fail",
              success_dir="output/success",
              err_dir="output/err",
              bug_dir="output/bug") -> List[Tuple[bool, ArgumentMapping]]:
    """ Map the arguments of api A to api B's arguments.

    Returns None if matching fails. """
    def verify_wrapper(api_A,
                       api_B,
                       argmat: ArgumentMapping,
                       neq_dir,
                       fail_dir,
                       success_dir,
                       err_dir,
                       bug_dir) -> bool:
        for _ in range(num_verify):
            res = verify(api_A,
                         api_B,
                         argmat,
                         num_inputs,
                         equal_type,
                         fuzz,
                         neq_dir, fail_dir, success_dir, err_dir, bug_dir,
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
    
    api_A = TFAPI(A)
    api_B = TFAPI(B)
    if api_A.api_def == None:
        return None
    if api_B.api_def == None:
        return None
    if len(api_A.arg_defs) == 0 or len(api_B.arg_defs) == 0 or (api_A.is_class != api_B.is_class):
        return None
   
    argmaps = get_api_map(A, B)
    
    results = []
    for argmap in argmaps:
        results.append((verify_wrapper(api_A, api_B, argmap, neq_dir, fail_dir,
                        success_dir, err_dir, bug_dir), argmap))
    return results



def test_api(A: str,
             B: str,
             argmap,
             num_inputs=50,
             equal_type=1,
             fuzz=True,
             neq_dir="output/neq",
             fail_dir="output/fail",
             success_dir="output/success",
             err_dir="output/err",
             bug_dir="output/bug",
             maxcnt=None):
    """map the arguments of api A to api B's arguments"""

    os.makedirs(neq_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(bug_dir, exist_ok=True)
    api_A = TFAPI(A)
    api_B = TFAPI(B)
    results = verify(api_A, api_B, argmap, num_inputs, equal_type, fuzz, neq_dir,
                   fail_dir, success_dir, err_dir, bug_dir, maxcnt=maxcnt)
    return results
