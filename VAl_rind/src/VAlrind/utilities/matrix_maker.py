import numpy as np 

def matrix_maker(row , col):

    value = np.empty(shape = (len(row) , len(col)))

    for row_index in range(len(row)):

        for col_index in range(len(col)):

            value[row_index][col_index] = row[row_index] * col[col_index]

    return value