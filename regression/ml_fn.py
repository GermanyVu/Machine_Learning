import numpy as np
from inspect import currentframe

def printd(*arg):
    frameinfo = currentframe()
    print(frameinfo.f_back.f_lineno,":",arg)

def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    printd('shuffled' , shuffled_indices)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size] # slice [0:size]
    #printd('test indices' , test_indices)
    train_indices= shuffled_indices[test_set_size:]# slice [size: end]
    #printd('train_indices',train_indices)
    return data.iloc[train_indices],data.iloc[test_indices]
