import textdistance
import numpy as np
from classes.argument import Argument
import torch

class ArgDef:
    def __init__(self):
        self.name: str = ''
        self.is_optional: bool = False
        self.must_use_name: bool = False
        self.type: set = set()
        self.default_value: str = ''
        self.description: str = ''
        self.case: Argument = None
        self.record = None
        self.ignore: bool = False
        self.disable = False
    
    @staticmethod
    def new(record):
        arg = ArgDef()
        arg.name = record["name"]
        arg.is_optional = record["is_optional"]
        arg.must_use_name = record["must_use_name"]
        arg.type = set(record["type"])
        arg.default_value = record["default_value"]
        arg.description = record["description"]
        return arg
    
    def arg_similar(self, arg: 'ArgDef'):
        def name_wrapper(name):
            if name == "_input_tensor":
                return "input"
            else:
                return name
        name_sim = self.string_similar(name_wrapper(self.name), name_wrapper(arg.name))

        if len(self.type) == 0:
            type_sim = 0
        else:
            type_sim = len(self.type.intersection(arg.type)) / len(self.type)
        
        return name_sim + type_sim    

    def args_similar(self, args: list['ArgDef'], w_name=0.3, w_type=0.7):
        sims = []
        for arg in args:
            sims.append(self.arg_similar(arg, w_name, w_type))
        return list(sims)
    
    def perfect_match(self, arg: 'ArgDef'):
        return ArgDef.perfect_match_(self, arg)
    
    @staticmethod
    def similarity(argdefs_a: list['ArgDef'], argdefs_b: list['ArgDef']):
        use_position = len(argdefs_a) and len(argdefs_b) and argdefs_a[0].name != "_input_tensor" and argdefs_b[0].name != "_input_tensor"
        sim = []
        for idx_a, def_a in enumerate(argdefs_a):
            temp = []
            for idx_b, def_b in enumerate(argdefs_b):
                t = def_a.arg_similar(def_b)
                if use_position:
                    t += 1 - (abs(idx_a - idx_b) / max(len(argdefs_a), len(argdefs_b)))
                # print(def_a.name, def_b.name, t)
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
