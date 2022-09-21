# db
TF_API_DOCS_KEY = "tf_api_docs"
API_DEF_KEY = "api_def"


# API attribute keys
API_NAME_KEY = 'api_name' # tf.argsort
TYPE_KEY = 'type' # one of ["function", "class", "module", "method" ]
DESC_KEY = 'desc' # one-sentence description 
DECLARATION_KEY = 'dec' # full declaration
ALIAS_KEY = 'alias' # alias name
METHODS_KEY = 'methods' # list of methods
ATTIBUTE_KEY = 'attributes'
RAISE_KEY = 'raises'
ARGS_KEY = 'args' # ordered list of arguments
RETURN_KEY = 'return' # return value
DETAILED_DESC_KEY = 'detail_desc' # detailed description
CODE_KEY = 'code' # code snippet
# args attribute keys
ARG_NAME_KEY = 'name' # arg_name
ARG_TYPE_KEY = 'type' # arg_type
ARG_DESC_KEY = 'desc' # arg_desc
ARG_DEFAULT_VALUE_KEY = 'dft_value' # default value, None if not exist
ARG_OPTIONAL_KEY = 'is_optional'

API_INVOCATION_INPUT_KEY = "__input__"

# execution result
RES_KEY = "results"
RESULT_KEY = "res"
ERROR_KEY = "err"
ERR_ARG_KEY = "error_args"
ERR_CPU_KEY = "err_cpu"
ERR_GPU_KEY = "err_gpu"
RES_CPU_KEY = "res_cpu"
RES_GPU_KEY = "res_gpu"
ERR_HIGH_KEY = "err_high"
ERR_LOW_KEY = "err_low"
RES_HIGH_KEY = "res_high"
RES_LOW_KEY = "res_low"
TIME_LOW_KEY = "time_low"
TIME_HIGH_KEY = "time_high"


ERR_1 = "err_1"
ERR_2 = "err_2"
RES_1 = "res_1"
RES_2 = "res_2"