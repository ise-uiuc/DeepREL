import pymongo
from numpy.random import choice
from constant.keys import API_DEF_KEY, API_NAME_KEY, ARG_NAME_KEY, ARGS_KEY
from utils.utils import softmax, string_similar
"""
This file is the interfere with database
"""

class Database:
    signature_collection = "signature"
    similarity_collection = "similarity"
    argdef_collection = API_DEF_KEY

    def __init__(self) -> None:
        self.lib_name: str = ""
        self.API_args: dict = None
        self.API_defs: dict = None
        self.colname = None
        

    def database_config(self, host, port, database_name):
        self.DB = pymongo.MongoClient(host=host, port=port)[database_name]

    def index_name(self, api_name, arg_name):
        arg_names = self.DB[self.signature_collection].find_one({"api": api_name})["args"]
        for idx, name in enumerate(arg_names):
            if name == arg_name:
                return f"parameter:{idx}"
        return None

    def get_api_arg_names(self, api_name):
        api = self.DB[API_DEF_KEY].find_one({API_NAME_KEY: api_name})
        args = api[ARGS_KEY]
        arg_names = [a[ARG_NAME_KEY] for a in args]
        return arg_names
        

    def select_rand_over_db(self, api_name, arg_name):
        if api_name not in self.DB.list_collection_names():
            return None, False
        arg_names = self.get_api_arg_names(api_name)
        if arg_name.startswith("parameter:"):
            index = int(arg_name[10:])
            if index >= len(arg_names):
                return None, False
            arg_name = arg_names[index]

        sim_dict = self.DB[self.similarity_collection].find_one({
            "api": api_name,
            "arg": arg_name
        })
        if sim_dict == None:
            return None, False
        APIs = sim_dict["APIs"]
        probs = sim_dict["probs"]
        if len(APIs) == 0:
            return None, False
        target_api = choice(APIs, p=probs)
        idx_name = self.index_name(target_api, arg_name)
        if idx_name == None:
            return None, False
        select_data = self.DB[target_api].aggregate([{
            "$match": {
                "$or": [{
                    arg_name: {
                        "$exists": True
                    },
                }, {
                    idx_name: {
                        "$exists": True
                    }
                }]
            }
        }, {
            "$sample": {
                "size": 1
            }
        }])
        if not select_data.alive:
            # not found any value in the (target_api, arg_name)
            print(f"ERROR IN SIMILARITY: {target_api}, {api_name}")
            return None, False
        select_data = select_data.next()
        if arg_name in select_data.keys():
            return select_data[arg_name], True
        else:
            return select_data[idx_name], True

    def get_rand_record(self, api_name):
        record = self.DB[api_name].aggregate([{"$sample": {"size": 1}}])
        if not record.alive:
            print(f"NO SUCH API: {api_name}")
            assert(0)
        record = record.next()
        record.pop("_id")
        return record

    def get_all_records(self, api_name):
        """ Returns all invocation records for an API. """
        temp = self.DB[api_name].find({}, {"_id": 0}) # Exclude _id in results
        records = []
        for t in temp:
            records.append(t)
        return records
    
    def get_apidefs(self):
        records = self.DB[self.argdef_collection].find({}, {"_id": 0})
        return records

    def get_argdef(self, api_name):
        record = self.DB[self.argdef_collection].find_one({API_NAME_KEY: api_name}, {"_id": 0})
        if record == None:
            print(f"NO API_ARGS FOR: {api_name}")
            assert(0)
        return record[ARGS_KEY]
    
    def add_records(self, api_name, records):
        self.DB[api_name].insert_many(records)

    def add_record(self, api_name, record):
        self.DB[api_name].insert_one(record)

    def add_signature(self, api_name, argnames):
        data = [ {
            "api": api_name, "args": argnames
        }]
        self.DB[self.signature_collection].insert_many(data)
    
    def get_signature(self, api_name):
        return self.DB[self.signature_collection].find_one({"api": api_name})
    
    def find_document(self, collection_name, query_key, query_value):        
        col = self.DB[collection_name]
        query = {query_key: query_value}
        return col.find_one(query)
        
    def update_doc_value(self, collection_name, query_key, query_value, update_key, update_value):
        
        col = self.DB[collection_name]
        query = {query_key: query_value}
        newvalues = { "$set": { update_key: update_value } }
        
        col.update_one(query, newvalues)

    #region DB mutation related
    

    def load_API_args(self, api_file=None):
        import re
        if api_file == None:
            api_file = f'../data/{self.lib_name}/{self.lib_name}_APIdef.txt'
        API_def = {}
        API_args = {}
        with open(api_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip()
                API_name = line.split("(")[0]
                API_args_match = re.search("\((.*)\)", line)
                try:
                    API_args_text = API_args_match.group(1)
                except:
                    raise ValueError(line)
                if API_name not in API_def.keys():
                    API_def[API_name] = line
                    API_args[API_name] = API_args_text
            self.API_def = API_def
            self.API_args = API_args
   
    def _build_collection_name(self):
        self.colname = self.DB.list_collection_names()

    def query_argname(self, arg_name):
        '''
        Return a list of APIs with the exact argname
        '''
        if self.colname is None:
            self._build_collection_name()
        DB = self.DB
        if self.API_args == None:
            self.load_API_args()
        API_args = self.API_args
        def index_name(api_name, arg_name):
            arg_names = self.get_signature(api_name)["args"]
            for idx, name in enumerate(arg_names):
                if name == arg_name:
                    return f"parameter:{idx}"
            return None
        APIs = []
        for api_name in API_args.keys():
            # search from the database
            # if arg_name exists in the records of api_name, append api_name into APIs
            if api_name not in self.colname or arg_name not in API_args[api_name]:
                continue
            temp = DB[api_name].find_one({arg_name: {"$exists": True}})
            if temp == None:
                # since there are two forms of names for one argument, {arg_name} and parameter:{idx}
                # we need to check the parameter:{idx}
                idx_name = index_name(api_name, arg_name)
                if idx_name and DB[api_name].find_one({idx_name: {"$exists": True}}):
                    APIs.append(api_name)
            else:
                APIs.append(api_name)
        return APIs

    
    def write_API_signature(self):
        """
        API's signature will be stored in 'signature' collection with the form
            api: the name of api
            args: the list of arguments' names
        """
        DB = self.DB
        names = DB.list_collection_names()
        library_name = self.lib_name
        if self.API_args == None:
            self.load_API_args()
        API_args = self.API_args
        for api_name in names:
            if not api_name.startswith(library_name):
                continue
            if api_name not in API_args.keys():
                DB[self.signature_collection].insert_one({"api": api_name, "args": []})
                continue

            arg_names = []
            for temp_name in API_args[api_name].split(","):
                temp_name = temp_name.strip()
                if len(temp_name) == 0 or temp_name == "*":
                    continue
                if "=" in temp_name:
                    temp_name = temp_name[:temp_name.find("=")]
                arg_names.append(temp_name)
            DB[self.signature_collection].insert_one({
                "api": api_name,
                "args": arg_names
            })


    def similarAPI(self, API, argname):
        '''
        Return a list of similar APIs (with the same argname) and their similarities
        '''
        API_with_same_argname = self.query_argname(argname)
        if len(API_with_same_argname) == 0:
            return [], []
        probs = []
        original_def = self.API_def[API]
        for item in API_with_same_argname:
            to_compare = self.API_def[item]
            probs.append(string_similar(original_def, to_compare))
        prob_norm2 = softmax(probs)
        return API_with_same_argname, prob_norm2

    def write_similarity_for_api(self, api_name):
        DB = self.DB
        args = DB[API_DEF_KEY].find_one({API_NAME_KEY: api_name})[ARGS_KEY]
        arg_names = [x[ARG_NAME_KEY] for x in args]
        for arg_name in arg_names:
            APIs, probs = self.similarAPI(api_name, arg_name)
            sim_dict = {}
            sim_dict["api"] = api_name
            sim_dict["arg"] = arg_name
            sim_dict["APIs"] = APIs
            sim_dict["probs"] = list(probs)
            DB[self.similarity_collection].insert_one(sim_dict)

    def write_similarity(self):
        """
        Write the similarity of (api, arg) in 'similarity' with the form:
            api: the name of api
            arg: the name of arg
            APIs: the list of similar APIs
            probs: the probability list
        """
        library_name = self.lib_name
        names = self.DB.list_collection_names()
        for api_name in names:
            if not api_name.startswith(library_name):
                continue

            print(api_name)
            self.write_similarity_for_api(api_name)
    #endregion

    @staticmethod
    def get_api_list(DB, start_str):
        api_list = []
        for name in DB.list_collection_names():
            if name.startswith(start_str):
                api_list.append(name)
        return api_list


class TFDB(Database):
    def __init__(self) -> None:
        super().__init__()
        self.lib_name = 'tf'

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "tf.")
        return self.api_list

"""
Database for each library
NOTE:
You must config the database by using `database_config(host, port, name)` before use!!!
Like TFDatabase.database_config("127.0.0.1", 27109, "tftest")
"""
TFDatabase = TFDB()

