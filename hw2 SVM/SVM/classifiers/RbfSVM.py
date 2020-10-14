import numpy as np
import random
from dataloader import load_data
from Kernel import Kernel
from utils import plot_curves

class RbfSVM(object):
    def __init__(self, data, batch_size = 200, learning_rate = 1e-7, epochs = 50, reg_type = 2, reg_weight = 0.1):
        #初始化参数
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data
        self.N = len(self.x_train)
        self.alpha = np.random.randn(self.N, ) * 0.0001


        #定义各种超参数
        self.batch_size = batch_size #SGD的minibatch size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_type = 0 #0：不正则化 1:L1正则化 2：L2正则化
        self.reg_weight = reg_weight #正则化的weight
        self.kernel = Kernel()

    def run(self):
        '''
        描述：主函数
        参数：无
        返回：最优的参数alpha
        '''
        max_acc = 0
        self.kernel.init_sigma(self.x_train)
        self.kernel.init_K(self.x_train)
        loss_train_list = []
        acc_train_list = []
        loss_eval_list = []
        acc_eval_list = []
        best_alpha = self.alpha.copy()
        for i in range(self.epochs):
            loss_train, acc_train = self.train_or_eval("Train", self.x_train, self.y_train, i + 1)
            loss_eval, acc_eval = self.train_or_eval("Eval", self.x_test, self.y_test, i + 1)
            loss_train_list.append(loss_train)
            loss_eval_list.append(loss_eval)
            acc_train_list.append(acc_train)
            acc_eval_list.append(acc_eval)
            if acc_eval > max_acc:
                max_acc = acc_eval
                best_alpha = self.alpha.copy()
        self.alpha = best_alpha.copy()
        loss_test, acc_test = self.train_or_eval("Test", self.x_test, self.y_test, self.epochs)
        plot_curves(loss_train_list, loss_eval_list, acc_train_list, acc_eval_list)

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
            y_i = y[i]
            if y_i == 0:
                y_i = -1

            result = 0
            for j in range(self.N):
                y_j = self.y_train[j]
                x_j = self.x_train[j]
                if y_j == 0:
                    y_j = -1
                new_num = y_j * self.alpha[j] * self.kernel.get(x_i, x_j)
                result += new_num
            if result * y_i >= 0:
                total_right += 1
        acc = total_right / N
        loss, dalpha = self.SGD(X, y)
        if mode == 'Train':
            self.alpha -= self.learning_rate * dalpha
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
        dalpha = np.zeros(self.alpha.shape)  
        N = X.shape[0]
        D = X.shape[1]

        #随机sample一个minibatch
        indexs = np.random.choice(N, self.batch_size)

        #先求hinge loss次梯度和loss
        for i in range(self.batch_size):
            #读取x，y，注意把y转换成1，-1
            x_i = X[indexs[i]]
            y_i = y[indexs[i]]
            if(y_i == 0):
                y_i = -1

            #求判据
            sum_result = 0.0
            for j in range(N):
                current_result = self.alpha[j] * self.kernel.get(x_i, X[j])
                sum_result += current_result
            difference = sum_result * y_i

            if difference < 1:
                for j in range(N):
                    dj = y_i * self.kernel.get(x_i, X[j])
                    dalpha[j] -= dj
                loss += (1 - difference)

            
        dalpha = dalpha / self.batch_size
        loss = loss / self.batch_size

        #考虑K*alpha^T
        loss += np.matmul(self.alpha.T, np.matmul(self.kernel.K, self.alpha)) / 2
        dalpha += np.matmul(self.kernel.K, self.alpha.T)

        #正则化
        if self.reg_type == 2 :
            dalpha += 2 * self.reg_weight * self.alpha
        elif self.reg_type == 1: 
            for i in range(D):
                if self.alpha[i] > 0:
                    dalpha[i] += self.reg_weight
                elif self.alpha[i] < 0:
                    dalpha[i] -= self.reg_weight
        
        return loss, dalpha

if __name__ == "__main__":
    data = load_data()
    svm = RbfSVM(data)
    svm.run()