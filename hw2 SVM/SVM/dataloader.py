import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_data():
    '''
        描述：读取数据集并且切分
        参数：无
        返回：x_train, y_train, x_val, y_val, x_test, y_test
    '''
    diabetes = pd.read_csv('datasets/diabetes.csv')
    x_train, x_test, y_train, y_test = train_test_split(np.array(diabetes.loc[:, diabetes.columns != 'Outcome']), np.array(diabetes['Outcome']), stratify=np.array(diabetes['Outcome']), random_state=66)
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
    
    
    return X_train, y_train, X_val, y_val, X_test, y_test

