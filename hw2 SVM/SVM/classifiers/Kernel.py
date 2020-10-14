import numpy as np
import math

class Kernel(object):        
    def init_sigma(self, X):
        list_num = []
        for i in range(len(X)):
            for j in range(len(X)):
                diff = X[i] - X[j]
                dist = np.dot(diff, diff)
                list_num.append(dist)
        list_num.sort()
        length = len(list_num)
        if length % 2 == 1:
            self.sigma = list_num[length // 2]
        else: 
            self.sigma = (list_num[length // 2] + list_num[length // 2 + 1]) / 2

    def init_K(self, X):
        N = X.shape[0]
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i][j] = self.get(X[i], X[j])
        self.K = K

    def get(self, x, y):
        diff = x - y
        dist = np.dot(diff, diff)
        exp_up = - dist / 2 / self.sigma
        result = math.exp(exp_up)
        result += 1
        return result