
import textdistance
from re import S
import numpy as np

from utils.probability import do_type_mutation


def find_match_br(s, charac):
    cnt = 0
    for i, c in enumerate(s):
        if c in ['(', '[', '{']:
            cnt += 1
            _s += c
        elif c in [')', ']', '}']:
            cnt -= 1
            _s += c
        elif cnt == 0 and c == charac:
            return i
    return -1

def str_to_value(s: str):
    if s == "True":
        return True
    if s == "False":
        return False
    if s == "None":
        return None
    try:
        int_value = int(s)
        return int_value
    except: pass
    try:
        ft_value = float(s)
        return ft_value
    except: pass
    s = s.strip()
    if s == "[]": return []
    if s == "()": return ()
    if s == "": return None
    if s[0] == '[' and s[-1] == ']':               
        t = s[1:-1]
        lst = [str_to_value(x) for x in t.split(',')]
        return lst
    if s[0] == '(' and s[-1] == ')':
        t = s[1:-1]
        lst = [str_to_value(x) for x in t.split(',')]
        return tuple(lst)
    return s


def mean_norm(x):
    return (x - np.mean(x)) / (np.max(x) - np.min(x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

    
def string_similar(s1, s2):
    return textdistance.levenshtein.normalized_similarity(s1, s2)
