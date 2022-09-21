import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os.path import join
import tensorflow as tf
import time
import numpy as np

from classes.argument import Argument, ArgType
from classes.tf_api import TFAPI, TFArgument
from classes.library import Library
from classes.database import TFDatabase
from constant.enums import OracleType
from constant.keys import ERR_CPU_KEY, ERR_GPU_KEY, ERR_HIGH_KEY, ERR_LOW_KEY, RES_CPU_KEY, RES_GPU_KEY, TIME_HIGH_KEY, TIME_LOW_KEY

class TFLibrary(Library):
    def __init__(self, output_dir, diff_bound=1e-5, time_bound=10, time_thresold=1e-3) -> None:
        super().__init__(output_dir)
        self.diff_bound = diff_bound
        self.time_bound = time_bound
        self.time_thresold = time_thresold
    
    @staticmethod
    def run_code(code):
        results = dict()
        results[ERR_CPU_KEY] = None
        results[ERR_GPU_KEY] = None
        results[ERR_HIGH_KEY] = None
        results[ERR_LOW_KEY] = None
        error = None
        MARK_DONE_FLAG=False
        try:
            exec(code)
            MARK_DONE_FLAG = True
        except Exception as e:
            error = str(e)
        return results, error, MARK_DONE_FLAG
    
    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if isinstance(x, tf.Tensor):
            return ArgType.TF_TENSOR
        elif isinstance(x, tf.DType):
            return ArgType.TF_DTYPE
        else:
            return ArgType.TF_OBJECT

    
    @staticmethod
    def _eval_k(x):
        import keras.backend as K
        def K_eval(x):
            try:
                return K.get_value(K.to_dense(x))
            except:
                eval_fn = K.function([], [x])
                return eval_fn([])[0]
        
        # return K.eval(x)
        
        return tf.convert_to_tensor(x).numpy()
    @staticmethod
    def is_equal(x, y):
        x_type = TFArgument.get_type(x)
        y_type = TFArgument.get_type(y)
        if x_type != y_type:
            return False
        if x_type == ArgType.KERAS_TENSOR:
            return tf.math.equal(x, y)
        if x_type == ArgType.TF_TENSOR:
            np_x = x.numpy()
            np_y = y.numpy()
            if x.dtype.is_complex: 
                return True
            if isinstance(x, tf.RaggedTensor):
                return True
            try:
                if x.dtype.is_floating:
                    return tf.experimental.numpy.allclose(np_x, np_y, rtol=1e-3, atol=1e-4,equal_nan=True).numpy().all()
                elif x.dtype.is_integer:
                    return tf.experimental.numpy.equal(np_x, np_y).numpy().all()
            except:
                return False
            # not strictly equal
            return True
        elif x_type == ArgType.FLOAT:
            return abs(x - y) < 1e-5
        elif x_type in [ArgType.LIST, ArgType.TUPLE]:
            if len(x) != len(y):
                return False
            for i in range(len(x)):
                if TFLibrary.is_equal(x[i], y[i]) == False:
                    return False
            return True
        
        else:
            try:
                flag = x == y
            except:
                print(x, y)
                return False

            if isinstance(flag, np.ndarray):
                flag = flag.all()
            try:
                if flag:
                    pass
            except:
                flag = True
            return flag

def if_skip(api):
    skip_list = ["tf.keras.Input", "tf.keras.layers.Input"]
    if api in skip_list:
        return True
    skip_keyword = ["initializers", "tf.keras.applications."]
    for k in skip_keyword:
        if k in api:
            return True
    return False
