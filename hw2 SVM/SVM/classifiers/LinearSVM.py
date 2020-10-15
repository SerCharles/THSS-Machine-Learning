import numpy as np
import random


class LinearSVM(object):
    def __init__(self, data, batch_size = 200, learning_rate = 1e-6, epochs = 100, reg_type = 2, reg_weight = 1e-3, whether_print = True):
        #初始化参数
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data
        self.N = len(self.x_train)
        self.W = np.random.randn(9, ) * 0.000001


        #定义各种超参数
        self.batch_size = batch_size #SGD的minibatch size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_type = 0 #0：不正则化 1:L1正则化 2：L2正则化
        self.reg_weight = reg_weight #正则化的weight
        self.whether_print = whether_print

    def run(self):
        '''
        描述：主函数
        参数：无
        返回：loss_train_list, loss_eval_list, acc_train_list, acc_eval_list, loss_test, acc_test
        '''
        max_acc = 0
        best_W = self.W.copy()
        loss_train_list = []
        acc_train_list = []
        loss_eval_list = []
        acc_eval_list = []
        for i in range(self.epochs):
            loss_train, acc_train = self.train_or_eval("Train", self.x_train, self.y_train, i + 1)
            loss_eval, acc_eval = self.train_or_eval("Eval", self.x_val, self.y_val, i + 1)
            loss_train_list.append(loss_train)
            loss_eval_list.append(loss_eval)
            acc_train_list.append(acc_train)
            acc_eval_list.append(acc_eval)
            if acc_eval > max_acc:
                max_acc = acc_eval
                best_W = self.W.copy()
        self.W = best_W.copy()
        loss_test, acc_test = self.train_or_eval("Test", self.x_test, self.y_test, self.epochs)
        return loss_train_list, loss_eval_list, acc_train_list, acc_eval_list, loss_test, acc_test
        #plot_curves(loss_train_list, loss_eval_list, acc_train_list, acc_eval_list)



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
        N = X.shape[0]
        total_right = 0
        for i in range(N):
            x_i = X[i]
            x_i = x_i.tolist()
            x_i.append(1)
            x_i = np.array(x_i)
            y_i = y[i]
            if(y_i == 0):
                y_i = -1

            result = np.dot(self.W, x_i) * y_i
            if result >= 0:
                total_right += 1
        acc = total_right / N
        loss, dW = self.SGD(X, y)
        if mode == 'Train':
            self.W -= self.learning_rate * dW
        if self.whether_print:  
            print("{} Epoch:[{}/{}] Accuracy:{:.4f} Loss:{:.4f}".format(mode, epoch, self.epochs, acc, loss))
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

        #考虑前面那一项
        #W_front = np.array(self.W.tolist()[:8])
        loss += np.dot(self.W, self.W)
        #dW_front = self.W.tolist()[:8]
        #dW_front.append(0)
        #dW_front = np.array(dW_front)
        dW += self.W

        #正则化
        if self.reg_type == 2 :
            dW += 2 * self.reg_weight * self.W
        elif self.reg_type == 1: 
            for i in range(D):
                if self.W[i] > 0:
                    dW[i] += self.reg_weight
                elif self.W[i] < 0:
                    dW[i] -= self.reg_weight
        
        return loss, dW

