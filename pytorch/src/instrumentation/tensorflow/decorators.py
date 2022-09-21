from tensorflow.instrumentation.signature_handler import SignatureHandler
import os
import json
sighdl = SignatureHandler()

def isiterable(t):
    return isinstance(t, list) or isinstance(t, tuple)

def get_signature_for_tensors(t):
    return sighdl.get_var_signature(t)


def build_param_dict(*args, **kwargs):
    param_dict = dict()
    for ind, arg in enumerate(args):
        param_dict['parameter:%d' % ind] = sighdl.get_var_signature(arg)
    for key, value in kwargs.items():
        if key == 'name': continue
        param_dict[key] = sighdl.get_var_signature(value)
    param_dict = dict(param_dict)
    return param_dict




def dump_signature_of_class(klass, class_name, output_dir):
    if not hasattr(klass, '__call__'):
        return klass
    old_init = klass.__init__
    old_call = klass.__call__
    init_params = dict()


    def write_fn(func_name, params, input_signature, output_signature):
        out_fname = class_name + '.txt'
        params['input_signature'] = input_signature
        params['output_signature'] = output_signature
        with open(os.path.join(output_dir, out_fname), 'a') as f:
            f.write(json.dumps(dict(params)) + chr(10))

    def new_init(self, *args, **kwargs):
        nonlocal init_params
        try:
            init_params = build_param_dict(*args, **kwargs)
        except Exception as e:
            print(e.message)
        old_init(self, *args, **kwargs)

    def new_call(self, *inputs, **kwargs):
        nonlocal init_params

        input_signature = get_signature_for_tensors(inputs)
        outputs = old_call(self, *inputs, **kwargs)
        output_signature = get_signature_for_tensors(outputs)
        write_fn(self.__class__.__module__ + '.' + self.__class__.__name__, init_params, input_signature,
                 output_signature)
        return outputs

    klass.__init__ = new_init
    klass.__call__ = new_call
    return klass

from functools import wraps


def dump_signature_of_function(func, hint, output_dir):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import json
        import os


        def write_fn(func_name, params, input_signature, output_signature):
            params = dict(params)
            out_fname = func_name + '.txt'
            if input_signature is not None:
                params['input_signature'] = input_signature
            params['output_signature'] = output_signature
            with open(os.path.join(output_dir, out_fname), 'a') as f:
                f.write(json.dumps(dict(params)) + chr(10))

        outputs = func(*args, **kwargs)
        output_signature = get_signature_for_tensors(outputs)
        param_dict = build_param_dict(*args, **kwargs)
        write_fn(hint, param_dict, None, output_signature)
        return outputs

    if not callable(func):
        return func

    return wrapper
