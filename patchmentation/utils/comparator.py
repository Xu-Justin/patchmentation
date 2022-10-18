from enum import Enum
from functools import partial

def equal(object1, object2):
    return object1 == object2

def greater_than(object1, object2):
    return object1 > object2

def greater_equal(object1, object2):
    return object1 >= object2

def less_than(object1, object2):
    return object1 < object2

def less_equal(object1, object2):
    return object1 <= object2

class Comparator(Enum):
    Equal = partial(equal)
    GreaterThan = partial(greater_than)
    GreaterEqual = partial(greater_equal)
    LessThan = partial(less_than)
    LessEqual = partial(less_equal)
    
    def __call__(self, *args):
        return self.value(*args)
