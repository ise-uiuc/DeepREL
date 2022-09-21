from typing import List
from functools import WRAPPER_UPDATES
import inspect
import json
import random
from typing import Dict
from copy import deepcopy
import numpy as np
import tensorflow as tf
from numpy.random import choice, randint

from constant.keys import *
from classes.argument import ArgType, Argument
from classes.api import API
from termcolor import colored

from classes.argdef import ArgDef
from classes.database import TFDatabase
from classes.library_def import LibraryAPIDef, tf_lib_def

from utils.probability import do_type_mutation, do_select_from_db

class TFArgument(Argument):
    _int_values = [-16, -1, 0, 1, 16]
    _str_values = ["", "1", "sum", "same", "valid", "zeros"]
    _float_values = [0.0, 1.0, -1.0, 63.0, -63.0]
    _tensor_arg_dtypes = [ArgType.TF_TENSOR, ArgType.KERAS_TENSOR, ArgType.TF_VARIABLE]
    _dtypes = [
        tf.bfloat16, tf.bool, tf.complex128, tf.complex64, tf.double,
        tf.float16, tf.float32, tf.float64, tf.half,
        tf.int16, tf.int32, tf.int64, tf.int8,
        tf.uint8, tf.uint16, tf.uint32, tf.uint64,
    ]
    _support_types = [
        ArgType.TF_TENSOR, ArgType.TF_VARIABLE, ArgType.KERAS_TENSOR,
        ArgType.TF_DTYPE, ArgType.TF_OBJECT
    ]

    def __init__(self, value, type: ArgType, minv=0, maxv=0, shape=None, dtype=None) -> None:
        if isinstance(dtype, str):
            dtype = self.str_to_dtype(dtype)
        shape = self.shape_to_list(shape)

        super().__init__(value, type)
        self.minv = minv
        self.maxv = maxv
        self.shape = shape
        self.dtype = dtype

        self.name = '' # Can be 'parameter:id' or var_name
        self.var_name = '' # The variable name, e.g. "input", "input_cp"
        self.arg_name = '' # The official argument name in signature.

    @staticmethod
    def str_to_dtype(dt: str):
        dt = dt.strip().replace("_ref", "")
        if not dt.startswith("tf."):
            dt = "tf." + dt
        try:
            return eval(dt)
        except:
            return tf.float32

    @staticmethod
    def shape_to_list(shape): 
        if shape is None: return None   
        if not isinstance(shape, list):
            try:
                shape = shape.as_list()
            except:
                shape = list(shape)
            else:
                shape = list(shape)
        shape = [1 if x is None else x for x in shape]
        return shape

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if tf.is_tensor(x):
            if tf.keras.backend.is_keras_tensor(x):
                return ArgType.KERAS_TENSOR
            return ArgType.TF_TENSOR
        elif isinstance(x, tf.DType):
            return ArgType.TF_DTYPE
    
    def _shape_to_record(self, shape):
        s = []
        for i in shape:
            if i != None:
                s.append(int(i))
            else:
                s.append(i)
        return s
    
    def is_tensorlike(self) -> bool:
        return self.type in self._tensor_arg_dtypes

    def to_record(self):
        
        if self.type == ArgType.INT:
            return int(self.value)
        elif self.type == ArgType.STR:
            return str(self.value)
        elif self.type == ArgType.FLOAT:
            return float(self.value)
        elif self.type == ArgType.BOOL:
            return bool(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            v = []
            for x in self.value:
                v.append(x.to_record())
            if self.type == ArgType.TUPLE:
                return tuple(v)
            else:
                return v
        elif self.type in self._tensor_arg_dtypes:
            rec = dict()
            rec["type"] = "tensor"
            rec["shape"] = self._shape_to_record(self.shape)
            rec["dtype"] = self.dtype.name
            return rec
        elif self.type == ArgType.TF_DTYPE:
            return self.value.name
        elif self.type == ArgType.TF_OBJECT:
            rec = dict()
            rec["type"] = "tf_object"
            return rec
        elif self.type == ArgType.NULL:
            return None
        else:
            raise ValueError(self.type)
            assert (0)
        
    def mutate_value_random(self) -> None:
        """ Apply random value mutation. """
        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            self.value = self.mutate_bool_value(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            # Reduce the probability of mutating important args
            if self.name == "shape":
                if random.randint(0,20) >= 3: return 
            else:
                if random.randint(0,5) >= 2: return 
            for arg in self.value:
                arg.mutate_value_random()
        elif self.type in self._tensor_arg_dtypes:
            self.minv, self.maxv = self.random_tensor_value_range(self.dtype)
        elif self.type == ArgType.TF_DTYPE:
            self.value = TFArgument.mutate_dtype()
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            pass
        else:
            raise ValueError(self.type)
            assert (0)

    def if_mutate_shape(self):
        return random.random() < 0.3

    def if_mutate_shape_value(self):
        return random.random() < 0.3

    def if_expand_dim(self):
        return random.random() < 0.3

    def if_squeeze(self):
        return random.random() < 0.3

    def mutate_shape(self, old_shape):
        new_shape = old_shape

        # Change rank
        if self.if_expand_dim():
            new_shape.append(1)
        elif len(new_shape) > 0 and self.if_squeeze():
            new_shape.pop()
        # Change value
        for i in range(len(new_shape)):
            if self.if_mutate_shape_value():
                new_shape[i] = self.mutate_int_value(new_shape[i], minv=0)
               
        return new_shape

    def generate_value_random(self) -> None:

        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(0)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value("")
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(0.)
        elif self.type == ArgType.BOOL:
            self.value = self.mutate_bool_value(True)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            self.value = [TFArgument(1, ArgType.INT), TFArgument(1, ArgType.INT)]
        elif self.type in self._tensor_arg_dtypes:
            shape = [randint(1, 3), randint(1, 3)]
            dtype = choice([tf.int32, tf.float32, tf.float64])
            self.shape, self.dtype = shape, dtype
            self.value, self.minv, self.maxv = None, 0, 1
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TF_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            assert (0)

    def mutate_type(self) -> None:
        def if_mutate_primitive():
            return random.random() < 0.5

        def if_mutate_null():
            return random.random() < 0.5

        def if_mutate_type_random():
            return random.random() < 0.2

        if self.type == ArgType.NULL:
            if not if_mutate_null():
                return False

        if if_mutate_type_random():
            self.type = ArgType.NULL

        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]:
            if not if_mutate_primitive(): return False
            # change the type
            types = [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]
            types.remove(self.type)
            self.type = choice(types)
            # change the value
            if self.type == ArgType.INT:
                self.value = self.mutate_int_value(0)
            elif self.type == ArgType.FLOAT:
                self.value = self.mutate_float_value(0.0)
            elif self.type == ArgType.STR:
                self.value = self.mutate_str_value("")
            elif self.type == ArgType.BOOL:
                self.value = choice([True, False])
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            if random.random() < 0.01: 
                self.value = [] # with a probability return an empty list
            for arg in self.value:
                arg.mutate_type()
        elif self.type == ArgType.TF_TENSOR:
            dtype = choice(self._dtypes)
            shape = self.shape
            if self.if_mutate_shape():
                shape = self.mutate_shape(shape)
            self.shape, self.dtype = shape, dtype
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            new_type = choice(self._support_types + super()._support_types)
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    TFArgument(2, ArgType.INT),
                    TFArgument(3, ArgType.INT)
                ]
            elif new_type == ArgType.TF_TENSOR:
                self.shape = [2, 2]
                self.dtype = tf.float32

            if new_type != ArgType.NULL:
                try:
                    self.type = new_type
                    self.generate_value_random()
                except:
                    pass
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(TFArgument._dtypes)
        return True

    @staticmethod
    def if_mutate_int_random():
        return random.random() < 0.2

    @staticmethod
    def if_mutate_str_random():
        return random.random() < 0.1

    @staticmethod
    def if_mutate_float_random():
        return random.random() < 0.2

    
    def mutate_bool_value(self, value) -> bool:
        return choice([True, False])

    def mutate_int_value(self, value, minv=None, maxv=None) -> int:
        if TFArgument.if_mutate_int_random():
            value = choice(self._int_values)
        else:
            value += randint(-2, 2)
        if minv is not None:
            value = max(minv, value)
        if maxv is not None:
            value = min(maxv, value)
        return value
    
    def mutate_str_value(self, value) -> str:
        if TFArgument.if_mutate_str_random():
            return choice(self._str_values)
        return value

    def mutate_float_value(self, value) -> float:
        if TFArgument.if_mutate_float_random():
            return choice(self._float_values)
        else:
            return value + randint(-16, 16)

    @staticmethod
    def mutate_dtype() -> tf.dtypes.DType:
        return choice(TFArgument._dtypes)


    @staticmethod
    def random_tensor_value_range(dtype):
        assert isinstance(dtype, tf.dtypes.DType)
        minv = 0
        maxv = 0
        if dtype.is_floating or dtype.is_complex or dtype == tf.string or dtype == tf.bool:
            pass
        elif dtype.name == "int64" or dtype.name == "uint64":
            minv = 0
            maxv = (1 << 16)
            if random.random() < 0.9:
                maxv = random.randint(0, 2)
        elif "uint" in dtype.name or "int" in dtype.name:
            minv = dtype.min if random.random() < 0.1 else 0
            maxv = dtype.max - 2 if random.random() < 0.1 else random.randint(0, 2)
            if maxv == tf.uint64.max:
                maxv = min(maxv, tf.int64.max - 2)
        else:
            try:
                minv = dtype.min
                maxv = dtype.max
            except Exception as e:
                minv, maxv = 0, 0
        return minv, maxv

    def to_code_tensor(self, var_name):
        dtype = self.dtype
        
        shape = self.shape
        if dtype is None:
            assert (0)
        code = ""
        var_tensor_name = f"{var_name}_tensor"
        if dtype.is_floating:
            code += "%s = tf.random.uniform(%s, dtype=tf.%s)\n" % (var_tensor_name, shape, dtype.name)
        elif dtype.is_complex:
            ftype = "float64" if dtype == tf.complex128 else "float32"
            code += "%s = tf.complex(tf.random.uniform(%s, dtype=tf.%s)," \
                    "tf.random.uniform(%s, dtype=tf.%s))\n" % (var_tensor_name, shape, ftype, shape, ftype)
        elif dtype == tf.bool:
            code += "%s = tf.cast(tf.random.uniform(" \
                   "%s, minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)\n" % (var_tensor_name, shape)
        elif dtype == tf.string:
            code += "%s = tf.convert_to_tensor(np.ones(%s, dtype=str))\n" % (var_tensor_name, shape)
        elif dtype in [tf.int32, tf.int64]:
            code += "%s = tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.%s)\n" \
                % (var_tensor_name, shape, self.minv, self.maxv + 1, dtype.name)
        else:
            code += "%s = tf.saturate_cast(" \
                "tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.int64), " \
                "dtype=tf.%s)\n" % (var_tensor_name, shape, self.minv, self.maxv + 1, dtype.name)
        code += f"{var_name} = tf.identity({var_tensor_name})\n"
        self.var_name = var_name
        return code

    def to_code_keras_tensor(self, var_name):
        return self.to_code_tensor(var_name)


    def random_tensor_value(self):
        
        dtype = self.dtype
        shape = self.shape

        if (not isinstance(self.shape, list)) or \
            not all([x >= 0 for x in self.shape]):
            raise ValueError(f"shape {self.shape} is invalid")

        tensor = tf.random.uniform(shape=[2,2], dtype=tf.float32)
        if dtype.is_floating:
            tensor = tf.random.uniform(shape, dtype=dtype)
        elif dtype.is_complex:
            ftype = "float64" if dtype == tf.complex128 else "float32"
            tensor = tf.complex(tf.random.uniform(shape, dtype=ftype),
                tf.random.uniform(shape, dtype=ftype))
        elif dtype == tf.bool:
            tensor = tf.cast(tf.random.uniform(shape, minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)
        elif dtype == tf.string:
            tensor = tf.convert_to_tensor(np.ones(shape, dtype=str))
        else:
            try:
                tensor = tf.saturate_cast(
                    tf.random.uniform(shape, minval=self.minv, maxval=self.maxv+1, dtype=tf.int64), dtype=dtype)
            except:
                raise ValueError(f"Unrecog {dtype}")
        return tensor


    def to_code(self, var_name) -> str:
        if var_name == "":
            raise ValueError(f"var name is empty for {self.name}")
        self.var_name = var_name
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}")
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, self.value.name)
        elif self.type in self._tensor_arg_dtypes:
            code = ""
            if self.type == ArgType.TF_TENSOR:
                code = self.to_code_tensor(var_name)
            elif self.type == ArgType.TF_VARIABLE:
                code = self.to_code_tensor(var_name)
                code += "%s = tf.Variable(%s)\n" % (var_name, var_name)
            elif self.type == ArgType.KERAS_TENSOR:
                code = self.to_code_keras_tensor(var_name)
            return code
        return super().to_code(var_name)


    def mutate_value(self):
        self.mutate_value_random()

    @staticmethod
    def generate_arg_from_signature(signature):
        # Check for bool first, otherwise it will be recognized as int.
        if isinstance(signature, bool):
            return TFArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return TFArgument(signature, ArgType.INT)
        if isinstance(signature, float):
            return TFArgument(signature, ArgType.FLOAT)
        if isinstance(signature, str):
            return TFArgument(signature, ArgType.STR)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TFArgument.generate_arg_from_signature(elem))
            return TFArgument(value, ArgType.LIST)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TFArgument.generate_arg_from_signature(elem))
            return TFArgument(value, ArgType.TUPLE)

        if (not isinstance(signature, dict)):
            return TFArgument(None, ArgType.NULL)

        if "type" not in signature and "Label" not in signature:
            # It is a dictionary...
            return TFArgument(None, ArgType.NULL)

        label = signature["type"] if "type" in signature else signature["Label"]

        if label == "tf_object":
            if "class_name" not in signature:
                return TFArgument(None, ArgType.TF_OBJECT)

            if signature["class_name"] == "tensorflow.python.keras.engine.keras_tensor.KerasTensor" or \
                signature["class_name"] == "tensorflow.python.ops.variables.RefVariable":
                dtype = signature["dtype"]
                shape = signature["shape"]
                dtype = TFArgument.str_to_dtype(dtype)
                minv, maxv = TFArgument.random_tensor_value_range(dtype)
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)
            if signature["class_name"] == "tensorflow.python.framework.dtypes.DType":
                name = signature["to_str"].replace("<dtype: '", "").replace("'>", "")
                value = eval("tf." + name)
                return TFArgument(value, ArgType.TF_DTYPE)
            try:
                value = eval(signature.class_name)
            except:
                value = None
            return TFArgument(value, ArgType.TF_OBJECT)
        if label == "raw":
            try:
                value = json.loads(signature['value'])
            except:
                value = signature['value']
                pass
            if isinstance(value, int):
                return TFArgument(value, ArgType.INT)
            if isinstance(value, str):
                return TFArgument(value, ArgType.STR)
            if isinstance(value, float):
                return TFArgument(value, ArgType.FLOAT)
            if isinstance(value, tuple):
                tuple_value = []
                for elem in value:
                    tuple_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(tuple_value, ArgType.TUPLE)
            if isinstance(value, list):
                list_value = []
                for elem in value:
                    list_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(list_value, ArgType.LIST)

        if label == "tuple":
            raise ValueError("Wrong signature " + str(signature))
        if label == "list":
            raise ValueError("Wrong signature " + str(signature))
        if label in ["tensor", "KerasTensor", "variable", "nparray"]:
            if not ('shape' in signature.keys()
                    and 'dtype' in signature.keys()):
                raise Exception('Wrong signature {0}'.format(signature))
            shape = signature['shape']
            dtype = signature["dtype"]
            dtype = TFArgument.str_to_dtype(dtype)

            if isinstance(shape, (list, tuple)):
                minv, maxv = TFArgument.random_tensor_value_range(dtype)
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)
            else:
                minv, maxv = 0, 1
                shape = [1, ]  
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)

        return TFArgument(None, ArgType.NULL)

class TFAPI(API):

    def __init__(self, api_name) -> None:
        """ Class for TensorFlow APIs.

        Args:
            args: Dict[str, TFArgument]. Stores the argument instances. 
            api_def: LibraryAPIDef
            arg_defs: List[ArgDef]
            is_class: bool
        """
        super().__init__(api_name)
        self.api_def:LibraryAPIDef = deepcopy(tf_lib_def.get_api(api_name))
        if self.api_def == None:
            print(f"[UNRECOGNISED API] {api_name}")
            self.arg_defs = []
            return 
        self.arg_defs = self.api_def.arg_defs
        self.is_class = self.api_def.is_class()
        # Initialize argment dict.
        self.args = {}
        for a in self.arg_defs:
            if a.is_optional:
                self.args[a.name] = TFArgument.generate_arg_from_signature(
                        a.default_value)
                self.args[a.name].var_name = a.name
                self.args[a.name].arg_name = a.name
            else:
                self.args[a.name] = None


    def get_arg_name(self, ind:int) -> str:
        arg_name = self.api_def.index2name(ind)
        return arg_name

    def get_arg(self, ind:int) -> TFArgument:
        arg_name = self.api_def.index2name(ind)
        if arg_name not in self.args:
            raise Exception(f"{arg_name} is not in the arg dict of {self.api}")
        return self.args[arg_name]

    def set_arg(self, ind:int, arg:TFArgument):
        arg_name = self.api_def.index2name(ind)
        self.args[arg_name] = deepcopy(arg)


    def get_invocation(self, record=None, debug=False) -> bool:
        """ Load record into self.args.

        Note that optional argument will be automatically created with default value. """
        for arg_def in self.arg_defs:
            arg_def.case = None
        if record == None:
            record = TFDatabase.get_rand_record(self.api)
        self.args = self.generate_args_from_record(record)
        

        # Add optional arguments to the argument list.
        for a in self.arg_defs:
            if a.is_optional and a.name not in self.args:
                self.args[a.name] = TFArgument.generate_arg_from_signature(
                        a.default_value)
                self.args[a.name].name = a.name
                self.args[a.name].arg_name = a.name
                self.args[a.name].var_name = a.name
                a.case = self.args[a.name]
            elif a.is_optional and a.name in self.args:
                a.case = self.args[a.name]
            elif a.name in self.args:
                a.case = self.args[a.name]
            else:
                return False
        return True


    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = TFDatabase.select_rand_over_db(self.api, arg_name)
                if success:
                    new_arg = TFArgument.generate_arg_from_signature(new_arg)
                    self.args[arg_name] = new_arg
                    new_arg.name = arg.name
                    new_arg.var_name = arg.var_name
                    new_arg.arg_name = arg.arg_name
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def info(self):
        print("INFO for ", self.api)
        for key, arg in self.args.items():
            if arg == None:
                print(f"  {key}: None")
            else:
                print(f"  {key}: {arg.name}, {arg.var_name}, {arg.arg_name}")

    def to_record(self) -> dict:
        record = {}
        for arg_name, arg in self.args.items():

            arg_rec = arg.to_record()
            if arg_name == API_INVOCATION_INPUT_KEY:
                # If arg is a invocation input:
                record['input_signature'] = arg_rec
            else:            
                # If arg is a normal argument:
                record[arg_name] = arg_rec
        return record
             

    def find_arg_with_name(self, arg_name) -> ArgDef:
        for arg_def in self.arg_defs:
            if arg_def.name == arg_name:
                return arg_def
        return None

    def generate_args_from_record(self, record: dict) -> Dict[str, TFArgument]:

        def generate_args_from_signatures(signatures):
            if isinstance(signatures, dict):
                if signatures['Label'] == 'list':
                    s = signatures['value']
                    if isinstance(s, list):
                        signatures = s
            args = []
            if signatures == None:
                return args
            for signature in signatures:
                x = TFArgument.generate_arg_from_signature(signature)
                args.append(x)
            return args

        args = {}
        for key in record.keys():
            if key == "input_signature":
                value = generate_args_from_signatures(record[key])
                args[API_INVOCATION_INPUT_KEY] = TFArgument(value, ArgType.LIST)
                args[API_INVOCATION_INPUT_KEY].name = key
                args[API_INVOCATION_INPUT_KEY].arg_name = API_INVOCATION_INPUT_KEY
                args[API_INVOCATION_INPUT_KEY].var_name = API_INVOCATION_INPUT_KEY
            elif key != "output_signature":
                argname = ''
                if "parameter:" in key:
                    ind = int(key[10:])
                    argname = self.get_arg_name(ind)
                else:
                    argname = key
                    if argname not in self.args:
                        pass
                        # **kwargs allows it.
                        # raise Exception(f"{argname} is not in key of {self.api}")
                args[argname] = TFArgument.generate_arg_from_signature(record[key])
                args[argname].name = key
                args[argname].arg_name = argname
                args[argname].var_name = argname
        return args

    def get_arg_by_names(self, argname, argname2):
        for k, a in self.args.items():
            if k in [argname, argname2]:
                return a
        return None

    def to_code(self, prefix="arg", res="res",
            skip_arg_to_code=False, **kwargs) -> str:
        arg_code = ""
        arg_str = ""
        count = 1

        for idx, arg_def in enumerate(self.arg_defs):
            arg = self.get_arg_by_names(arg_def.name, "parameter:"+str(idx))
            if arg_def.name == "**kwargs" and \
                isinstance(arg, dict) and len(arg) > 0:
                for key, argitem in arg.items():
                    if not skip_arg_to_code:
                        arg_code += argitem.to_code(key)
                    arg_str += f"{key}={key}, "
            elif arg!= None and arg_def.name != "input_signature":
                arg_case = arg

                if not skip_arg_to_code:
                    arg_code += arg_case.to_code(arg_case.var_name)

                # for the *shape, arg_def.name == *shape, arg_case.name == shape
                if arg_def.name in [API_INVOCATION_INPUT_KEY]:
                    pass
                elif arg_def.is_optional or 'raw_ops' in self.api:
                    arg_str += f"{arg_def.name}={arg.var_name}, "
                else:
                    arg_str += f"{arg.var_name}, "

        res_code = "tf.random.set_seed(42)\n"
        if self.is_class:
            res_code += f"{prefix}_class = {self.api}({arg_str})\n"

            if API_INVOCATION_INPUT_KEY in self.args.keys():
                arg_name = "__input__"
                if not skip_arg_to_code:
                    arg_code += self.args[API_INVOCATION_INPUT_KEY].to_code(
                        arg_name)
                res_code += f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code += f"{res} = {self.api}({arg_str})\n"
        invocation = self._to_invocation_code(arg_code, res_code, **kwargs)
        return invocation


    def _to_res_code(self, res_name, arg_str, input_name=None, prefix="arg"):
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            if input_name:
                res_code += f"{RES_KEY}[\"{res_name}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res_name}\"] = {self.api}({arg_str})\n"
        return res_code

    def _to_invocation_code(self, arg_code, res_code, use_try=False, err_name="", 
        wrap_device=False, device_name="", time_it=False, time_var="") -> str:
        if time_it:
            res_code = res_code + self.wrap_time(res_code, time_var)
        code = arg_code + res_code
        inv_code = code
        if wrap_device:
            inv_code = self.wrap_device(inv_code, device=device_name)
        if use_try:
            inv_code = self.wrap_try(inv_code, error_var=err_name)
        return inv_code

    @staticmethod
    def wrap_try(code:str, error_var) -> str:
        wrapped_code = "try:\n"
        if code.strip() == "":
            code = "pass"
        wrapped_code += API.indent_code(code)
        wrapped_code += f"except Exception as e:\n  {RES_KEY}[\"{error_var}\"] = \"Error:\"+str(e)\n"
        return wrapped_code

    @staticmethod
    def wrap_device(code:str, device) -> str:
        device_code = f"with tf.device('/{device}'):\n" + API.indent_code(code)
        return device_code

    @staticmethod
    def wrap_time(code:str, time_var) -> str:
        wrapped_code = "t_start = time.time()\n"
        wrapped_code += code
        wrapped_code += "t_end = time.time()\n"
        wrapped_code += f"{RES_KEY}[\"{time_var}\"] = t_end - t_start\n"
        return wrapped_code
    @staticmethod
    def invocation_code(res, error_res, res_code, use_try):
        code = ""
        if use_try:
            # specified with run_and_check function in relation_tools.py
            if error_res == None:
                error_res = res
            temp_code = "try:\n"
            temp_code += API.indent_code(res_code)
            temp_code += f"except Exception as e:\n  {error_res} = \"ERROR:\"+str(e)\n"
            res_code = temp_code

        code += res_code
        return code


