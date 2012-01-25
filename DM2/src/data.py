import csv
import numpy as np

def libras_movement():
    """
    Fetches the Libras Movement dataset

    Returns
    -------
        X, Y
    """
    dataset = csv.reader(open('data/movement_libras.data', 'r'))
    X = []
    Y = []
    for element in dataset:
        X.append(element[:-1])
        Y.append(element[-1])
    return np.array(X).astype('float'), np.array(Y).astype('float')
