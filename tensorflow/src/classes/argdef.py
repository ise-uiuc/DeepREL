   
from typing import Dict, List
import textdistance
import numpy as np
from constant.keys import *
from classes.argument import Argument


class ArgDef:
    def __init__(self):
        self.name: str = ''
        self.index: int = -1
        self.is_optional: bool = False
        self.type: set = set()
        self.default_value: str = ''
        self.description: str = ''
        self.case: Argument = None
        self.record = {}
        self.ignore: bool = False
    
    @staticmethod
    def new(record:Dict, index:int) -> 'ArgDef':
        arg = ArgDef()
        if ARG_DEFAULT_VALUE_KEY not in record:
            arg.name = record['name']
            arg.is_optional = record['is_optional']
            arg.type = set(record['type'])
            arg.default_value = record['default_value']
            arg.description = record['description']
            return arg
        

        arg.name = record[ARG_NAME_KEY]
        arg.index = index
        arg.is_optional = record[ARG_OPTIONAL_KEY]
        if arg.name in ["*args", "**kwargs"] :
            arg.is_optional = True
        arg.type = set() if ARG_TYPE_KEY not in record else set(record[ARG_TYPE_KEY])
        arg.default_value = record[ARG_DEFAULT_VALUE_KEY]
        # If the default strings contain extra quotes, remove them.
        if isinstance(arg.default_value, str):
            s = arg.default_value
            if s[0] == '\'' and s[-1] == '\'':
                s = s[1:-1]
            elif s[0] == '"' and s[-1] == '"':
                s = s[1:-1]
            arg.default_value = s
        if ARG_DESC_KEY in record:
            arg.description = record[ARG_DESC_KEY]
        return arg
    
    def arg_similar(self, arg: 'ArgDef', max_num_args, w_name=1.0, w_type=1.0, w_pos=1.0):
        def name_wrapper(name):
            if name == "_input_tensor":
                return "input"
            else:
                return name
        name_sim = self.string_similar(name_wrapper(self.name), name_wrapper(arg.name))
        if len(self.type) == 0 or len(arg.type) == 0:
            type_sim = 0.5
        else:
            type_sim = len(self.type.intersection(arg.type)) / len(self.type)
        

        pos_sim = 1.0 - abs(self.index - arg.index) / max_num_args

        # Combined similarity is the sum of all three.
        return name_sim + type_sim + pos_sim

    def args_similar(self, args: List['ArgDef'], max_num_args):
        sims = []
        for arg in args:
            sims.append(self.arg_similar(arg, max_num_args))
        return list(sims)
    
    def perfect_match(self, arg: 'ArgDef'):
        return ArgDef.perfect_match_(self, arg)
    
    @staticmethod
    def similarity(argdefs_a: List['ArgDef'], argdefs_b: List['ArgDef'], verbose=True):
        sim = []
        max_num_args = max(len(argdefs_a), len(argdefs_b))
        for def_a in argdefs_a:
            temp = []
            for def_b in argdefs_b:
                t = def_a.arg_similar(def_b, max_num_args)
                if verbose:
                    print(def_a.name, def_b.name, t)
                temp.append(t)
            sim.append(temp)

        return sim

    @staticmethod
    def perfect_match_(arg1: 'ArgDef', arg2: 'ArgDef'):
        flag = (arg1.name == arg2.name) and (arg1.type.intersection(arg2.type))
        return flag

    @staticmethod
    def string_similar(s1, s2):
        return textdistance.levenshtein.normalized_similarity(s1, s2)
