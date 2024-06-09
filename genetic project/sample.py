import sys
import numpy
from itertools import filterfalse
import copy
import random
import time
import sys
import plotly.figure_factory as ff
import datetime
import numpy as np
import random
from tqdm import tqdm

def removeFromList(parent, list):
    seen = set()
    seen_add = seen.add
    return [x for x in parent if not (x in list or seen_add(x))]
def main():
    a = [3,1,2,1,2,3,2,3,1]
    b = [3, 2, 1, 2, 1, 1, 3, 3, 2]
    c = a[2:6]
    d = removeFromList(b,c)
    print(c)
    print(d)
    result = []
    result.extend(d[:2])
    result.extend(c)
    result.extend(d[2:])
    print(result)

if __name__ == '__main__':
    main()
