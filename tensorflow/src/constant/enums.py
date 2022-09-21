
from enum import Enum, IntEnum
class OracleType(IntEnum):
    CRASH = 1
    CUDA = 2
    PRECISION = 3

    
class APIRepresentationIDMode(IntEnum):
  API_NAME = 1
  API_DESC = 2
  API_DEF = 3

  
class EqualType(Enum):
    VALUE = 1
    STATUS = 2