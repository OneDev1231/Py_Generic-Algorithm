import numpy as np
import math
import random
## Generate quantum population
def quantum_population(n, theta):
    log_n = int(math.log(n, 2) + 1)
    zero_string = "0" * n
    zero_string = zero_string * log_n
    array = [[0.70710678118 for _ in range(log_n * n)]]*2
    for i in range(len(array[0])):
        #this is the corrected part
        #if you use matmul in _quantum_rotation function, all params should be np.array()
        column = np.array([array[0][i], array[1][i]])
        rotated_column = _quantum_rotation(column, theta)
        array[0][i] = rotated_column[0]
        array[1][i] = rotated_column[1]
    modified_string = _modify_string_to_valid(zero_string, array)
    decimal_values = _convert_to_decimal(modified_string, n)
    rearranged = []
    for i in range(len(decimal_values)):
        if decimal_values[i] not in rearranged:
            rearranged.append(decimal_values[i])
        else:
            continue
    for i in range(1, len(decimal_values) + 1):
        if i not in rearranged:
            rearranged.append(i)
    return rearranged

def _quantum_rotation(column, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    rotated_column = np.matmul(R, column)
    return rotated_column

def _modify_string_to_valid(string, array):
    modified_string = ""
    for i in range(len(string)):
        r = random.random()
        if array[1][i] ** 2 > r:
            modified_string += "1"
        else:
            modified_string += "0"
    return modified_string

def _convert_to_decimal(string, n):
    log_n = int(math.log(n, 2) + 1)
    decimal_values = []
    for i in range(0, len(string), log_n):
        decimal_value = int(string[i:i+log_n], 2) % n
        decimal_values.append(decimal_value+1)
    return decimal_values