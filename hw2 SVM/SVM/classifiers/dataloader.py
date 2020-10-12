import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_data():
    '''
        描述：读取数据集并且切分
        参数：无
        返回：x_train, x_test, y_train, y_test
    '''
    diabetes = pd.read_csv('svm/datasets/diabetes.csv')
    x_train, x_test, y_train, y_test = train_test_split(np.array(diabetes.loc[:, diabetes.columns != 'Outcome']), np.array(diabetes['Outcome']), stratify=np.array(diabetes['Outcome']), random_state=66)
    return x_train, x_test, y_train, y_test
