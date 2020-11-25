import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_diabetes
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_data(norm):
    '''
        描述：读取数据集并且切分
        参数：==是否归一化
        返回：x_train, y_train, x_val, y_val, x_test, y_test
    '''
    data_dir = os.path.join(os.path.abspath('./'), 'datasets','diabetes.csv')
    diabetes = pd.read_csv(data_dir)
    x_train, x_test, y_train, y_test = train_test_split(np.array(diabetes.loc[:, diabetes.columns != 'Outcome']), np.array(diabetes['Outcome']), stratify=np.array(diabetes['Outcome']), random_state=66)
    
    #y如果是0改成-1
    for i in range(len(y_train)):
        if y_train[i] == 0:
            y_train[i] = -1
    for i in range(len(y_test)):
        if y_test[i] == 0:
            y_test[i] = -1 



    # Split the data into train, val, and test sets. In addition we will
    # create a small development set as a subset of the training data;
    # we can use this for development so our code runs faster.
    num_training = 480
    num_validation = 96 
    num_test = 160
    num_dev = 32

    # Our validation set will be num_validation points from the original
    # training set.
    mask = range(num_training, num_training + num_validation)
    X_val = x_train[mask]
    y_val = y_train[mask]

    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = x_train[mask]
    y_train = y_train[mask]

    # We will also make a development set, which is a small subset of
    # the training set.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = x_train[mask]
    y_dev = y_train[mask]

    # We use the first num_test points of the original test set as our
    # test set.
    mask = range(num_test)
    X_test = x_test[mask]
    y_test = y_test[mask]
    
    #x归一化
    if norm:
        mean = np.mean(X_train, axis = 0)
        std = np.std(X_train, axis = 0)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

    
    return X_train, y_train, X_val, y_val, X_test, y_test
