
from enum import Enum
import inspect
import json
import os
from pathlib import Path

from classes.argdef import ArgDef
from constant.enums import APIRepresentationIDMode, OracleType
from classes.database import TFDatabase
from constant.keys import ALIAS_KEY, API_DEF_KEY, API_NAME_KEY, ARG_DEFAULT_VALUE_KEY, ARG_NAME_KEY, ARG_OPTIONAL_KEY, ARGS_KEY, DECLARATION_KEY, DESC_KEY, DETAILED_DESC_KEY, ERR_1, ERR_2, ERR_CPU_KEY, ERR_GPU_KEY, RETURN_KEY, TYPE_KEY
from constant.paths import tf_api_json_dir, tf_data_dir, all_tf_symbol_file, standard_tf_symbol_file, func_tf_symbol_file
from utils.loader import json_load, load_api_names, load_data
from utils.printer import dump_data
from utils.utils import str_to_value
from typing import Dict, List
from functools import reduce
from re import finditer


class APIType(Enum):
    CLASS: 1
    FUNCTION: 2
    CLASS_METHOD: 3
    MODULE: 4

class LibraryAPIDef:


    def __init__(self, library_name=''):
        self.library = library_name
        self.id = ''
        self.name: str = ''  # Name of the library api, e.g. 'tf.math.multiply'
        self.typ: APIType = None
        self.alias: List[str] = [] # e.g. ['tf.multiply']
        self.description: str = ''  # e.g. 'Returns an element-wise x * y.'
        self.declaration: str = ''  # e.g. 'tf.math.multiply(x, y, name=None)'
        self.detail_desc: str = ''
        self.ret: str = ''
        self.arg_defs: List['ArgDef'] = []
        self.arg_declaration: List[Dict] = {} 
        self.arg_names: List[str] = []
        self.case: Dict = {}
        
    def from_json(self, json_obj:dict):
        
        self._dict = json_obj
        self.name = json_obj[API_NAME_KEY]
        if TYPE_KEY not in json_obj:
            # print(self.name)
            self.typ = "UNK"
        else:
            self.typ = json_obj[TYPE_KEY]
        self.description = json_obj[DESC_KEY]
        self.declaration = json_obj[DECLARATION_KEY]
        self.arg_declaration = json_obj[ARGS_KEY]
        if RETURN_KEY in json_obj: 
            self.ret = json_obj[RETURN_KEY]
        self.alias = json_obj[ALIAS_KEY]
        if DETAILED_DESC_KEY in json_obj:
            self.detail_desc = json_obj[DETAILED_DESC_KEY]
        self.arg_defs = []
        for arg_index, arg_record in enumerate(self.arg_declaration):
            arg_def = ArgDef.new(arg_record, arg_index)
            self.arg_defs.append(arg_def)
        self.arg_names = [a.name for a in self.arg_defs]
            

    def to_dict(self):
        api = {

            API_NAME_KEY: self.name,
            DESC_KEY: self.description, 
            ALIAS_KEY: self.alias,
            DECLARATION_KEY: self.declaration,
            ARGS_KEY: self.arg_declaration,
            DETAILED_DESC_KEY: self.detail_desc,
            RETURN_KEY: self.ret
        }
        return api

    def get_arg_by_index(self, index:int):
        assert index >= 0 and index < len(self.arg_declaration)
        return self.arg_declaration[index]
    def index2name(self, index:int):
        return self.get_arg_by_index(index)[ARG_NAME_KEY]

    def find_arg(self, arg_name) -> int:
        """ Returns the index of an argument `arg_name`. """
        for i, a in enumerate(self.arg_declaration):
            if a[ARG_NAME_KEY] == arg_name:
                return i
        return -1
    
    def set_case(self, case:Dict):
        self.case = case


    @staticmethod
    def is_arg_sig_optional(sig:Dict) -> bool:
        return sig[ARG_OPTIONAL_KEY]

    def is_optional(self, index:int) -> bool:
        arg = self.get_arg_by_index(index)
        return arg[ARG_OPTIONAL_KEY]
            
    def is_class(self) -> bool:
        return self.typ == "class"
    
    def get_value(self, arg_name):
        if arg_name in self.case:
            return self.case[arg_name]
        index = self.find_arg(arg_name)
        if index >= 0:
            arg = self.arg_declaration[index]
            is_opt = self.is_arg_sig_optional(arg)
            if is_opt:
                dft_value = arg[ARG_DEFAULT_VALUE_KEY]
                return dft_value
        else:
            raise ValueError(f"Argument {arg_name} not found for {self.name}")

from os.path import join
class LibraryDef():


    def __init__(self, name:str, data_dir:Path):
        self.name = name
        self.data_dir = data_dir
        self.apis: Dict[str, LibraryAPIDef] = dict()
        self.all_api_names: List[str] = []
        self.std_api_names: List[str] = []
        
    @staticmethod
    def run_code(code):
        import tensorflow as tf
        import numpy as np
        results = dict()
        results[ERR_1] = None
        results[ERR_2] = None
        error = None
        try:
            exec(code)
        except Exception as e:
            error = str(e)
        return results, error
        
    def load_api_names(self):
        if self.name == 'tf':
            api_names = [x[API_NAME_KEY] for x in TFDatabase.DB[API_DEF_KEY].find()]
            self.all_api_names = api_names
            self.std_api_names = api_names

    def load_apis(self, lazy=True):
        self.apis = dict()
        self.load_api_names()
        if lazy: 
            return 
        else:
            for api_name in self.std_api_names:
                self._load_api(api_name)

    def _load_api(self, api_name):
        api_data = [x for x in TFDatabase.DB[API_DEF_KEY].find({API_NAME_KEY: api_name})]
        if len(api_data) != 1:
            return
            
            
        api_data = api_data[0]
        api = LibraryAPIDef('tf')
        api.from_json(api_data)
        self.apis[api_name] = api
        

    def get_api(self, name):
        if name in self.std_api_names and name not in self.apis:
            self._load_api(name)
        if name in self.apis:
            return self.apis[name]
        else:
            print(f"{name} not in {self.name} library.")
            return None
    
    @staticmethod
    def generate_code():
        pass

    @staticmethod
    def write_to_dir(dir, code, maxcnt=None):
        filenames = os.listdir(dir)
        max_name = 0
        for name in filenames:
            max_name = max(max_name, int(name.replace(".py", "")))
        if maxcnt != None:
            if max_name > maxcnt:
                return ""
        new_name = str(max_name + 1) + ".py"
        with open(join(dir, new_name), "w") as f:
            f.write(code)
        return new_name


tf_lib_def = LibraryDef('tf', tf_data_dir)
