import numpy as np
import random


class LinearSVM(object):
    def __init__(self, data, batch_size = 256, learning_rate = 1e-5, epochs = 2000, reg_type = 2, reg_weight = 1e-3, whether_print = True, whether_average = False):
        #初始化参数
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data
        self.N = len(self.x_train)
        self.W = np.random.randn(9, ) * 0.000001
        self.current_W = self.W.copy()
        self.W_list = []

        #定义各种超参数
        self.batch_size = batch_size #SGD的minibatch size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_type = 0 #0：不正则化 1:L1正则化 2：L2正则化
        self.reg_weight = reg_weight #正则化的weight
        self.whether_print = whether_print
        self.whether_average = whether_average

    def run(self):
        '''
        描述：主函数
        参数：无
        返回：loss_train_list, loss_test, acc_test
        '''
        loss_train_list = []
        for i in range(self.epochs):
            loss_train, acc_train = self.train_or_eval("Train", self.x_train, self.y_train, i + 1)
            loss_train_list.append(loss_train)
            if (i + 1) % 100 == 0:
                loss_eval, acc_eval = self.train_or_eval("Eval", self.x_val, self.y_val, i + 1)
        loss_test, acc_test = self.train_or_eval("Test", self.x_test, self.y_test, self.epochs)
        return loss_train_list, loss_test, acc_test



    def train_or_eval(self, mode, X, y, epoch):
        '''
        描述：训练一个epoch
        参数：
            mode：训练还是测试
            X：(N, D)数据矩阵
            Y：(N, 1)数据向量
            epoch：当前epoch数
        返回：当前的loss和准确率
        '''
        loss = 0.0
        acc = 0
        N = X.shape[0]
        if (self.whether_print and epoch % 100 == 0) or mode == 'Test':
            result = self.predict(X)
            total_right = np.sum((result * y) > 0)
            acc = total_right / N
        if mode == 'Train':
            loss, dW = self.SGD(X, y)
            self.W -= self.learning_rate * dW
            self.W_list.append(self.W.copy())
            if self.whether_average:
                self.current_W = np.mean(np.array(self.W_list), axis = 0)
            else:
                self.current_W = self.W 

        if self.whether_print and epoch % 100 == 0:  
            if mode == 'Train':
                print("{} Epoch:[{}/{}] Accuracy:{:.4f} Loss:{:.4f}".format(mode, epoch, self.epochs, acc, loss))
            else: 
                print("{} Epoch:[{}/{}] Accuracy:{:.4f}".format(mode, epoch, self.epochs, acc))
        return loss, acc

    def SGD(self, X, y):
        '''
        描述：用SGD方法求得在hinge loss损失函数下alpha的梯度
        参数：
            X：(N, D)数据矩阵
            Y：(N, 1)数据向量
        返回：
            loss：hinge loss值
            dalpha：alpha的梯度
        '''
        # initialize the gradient and loss as zero
        loss = 0.0
        dW = np.zeros(self.W.shape)  
        N = X.shape[0]
        D = X.shape[1]

        #随机sample一个minibatch
        indexs = np.random.choice(N, self.batch_size)

        #先求hinge loss次梯度和loss
        for i in range(self.batch_size):
            #读取x，y，注意把x增加常数项，y转换成1，-1
            x_i = X[indexs[i]]
            x_i = x_i.tolist()
            x_i.append(1)
            x_i = np.array(x_i)
            y_i = y[indexs[i]]
            if(y_i == 0):
                y_i = -1

            #求判据
            classifier = y_i * np.dot(self.W, x_i)
            if classifier < 1:
                loss += (1 - classifier)
                dW -= (y_i * x_i)

        loss /= self.batch_size
        dW /= self.batch_size

        #正则化
        #最前面的 alpha*K*alpha啥的我没在之前考虑，和同学交流之后得知这个属于正则项的一部分，hinge loss才是真正的loss
        if self.reg_type == 2 :
            dW += 2 * self.reg_weight * self.W
            loss += self.reg_weight * np.dot(self.W, self.W)
        elif self.reg_type == 1: 
            for i in range(D):
                if self.W[i] > 0:
                    dW[i] += self.reg_weight
                    loss += self.reg_weight * self.W[i]
                elif self.W[i] < 0:
                    dW[i] -= self.reg_weight
                    loss -= self.reg_weight * self.W[i]

        
        return loss, dW

    def predict(self, X):
        '''
        描述：预测函数
        参数：X
        返回：输出结果y
        '''
        N = X.shape[0]
        total_right = 0
        y = np.zeros(N)
        for i in range(N):
            x_i = X[i]
            x_i = x_i.tolist()
            x_i.append(1)
            x_i = np.array(x_i)
            result = np.dot(self.current_W, x_i)
            y[i] = result
        return y
