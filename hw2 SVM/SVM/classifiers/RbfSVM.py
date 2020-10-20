import numpy as np
import random
from .Kernel import Kernel


class RbfSVM(object):
    def __init__(self, data, batch_size = 64, learning_rate = 1e-2, epochs = 2000, reg_type = 2, reg_weight = 1e-3, whether_print = True):
        #初始化参数
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data
        self.N = len(self.x_train)
        self.alpha = np.random.randn(self.N, ) * 0.0001
        self.b = random.random()* 0.0001

        self.alpha_average = self.alpha.copy()
        self.b_average = self.b

        #定义各种超参数
        self.batch_size = batch_size #SGD的minibatch size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_type = reg_type #0：不正则化 1:L1正则化 2：L2正则化
        self.reg_weight = reg_weight #正则化的weight
        self.kernel = Kernel()
        self.whether_print = whether_print


    def run(self):
        '''
        描述：主函数
        参数：无
        返回：loss_train_list, loss_eval_list, acc_train_list, acc_eval_list, loss_test, acc_test
        '''
        max_acc = 0
        self.kernel.init_sigma(self.x_train)
        self.kernel.init_K(self.x_train)
        loss_train_list = []
        acc_train_list = []
        loss_eval_list = []
        acc_eval_list = []
        best_alpha = self.alpha.copy()
        best_b = 0
        for i in range(self.epochs):
            loss_train, acc_train = self.train_or_eval("Train", self.x_train, self.y_train, i + 1)
            loss_eval, acc_eval = self.train_or_eval("Eval", self.x_val, self.y_val, i + 1)
            loss_train_list.append(loss_train)
            #loss_eval_list.append(loss_eval)
            #acc_train_list.append(acc_train)
            #acc_eval_list.append(acc_eval)
            '''if acc_eval > max_acc:
                max_acc = acc_eval
                best_alpha = self.alpha_average.copy()
                best_b = self.b_average
            '''
        #self.alpha_average = best_alpha.copy()
        #self.b_average = best_b
        loss_test, acc_test = self.train_or_eval("Test", self.x_test, self.y_test, self.epochs)
        return loss_train_list, loss_eval_list, acc_train_list, acc_eval_list, loss_test, acc_test

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
        if epoch % 100 == 0:
            result = self.predict(X)
            total_right = np.sum((result * y) > 0)
            acc = total_right / N
        if mode == 'Train':
            loss, dalpha, db = self.SGD(X, y)
            self.alpha -= self.learning_rate * dalpha
            self.b -= self.learning_rate * db
            self.alpha_average = (self.alpha_average * (epoch - 1) + self.alpha) / epoch
            self.b_average = (self.b_average * (epoch - 1) + self.b) / epoch
        else:
            loss = 0

        if self.whether_print and epoch % 100 == 0:  
            if mode == 'Train':
                print("{} Epoch:[{}/{}] Accuracy:{:.4f} Loss:{:.4f}".format(mode, epoch, self.epochs, acc, loss))
            else: 
                print("{} Epoch:[{}/{}] Accuracy:{:.4f}".format(mode, epoch, self.epochs, acc))
        return loss, 0

    def predict(self, X):
        N = X.shape[0]
        total_right = 0
        result = np.zeros(N)
        for i in range(N):
            x_i = X[i]
            the_sum = 0
            for j in range(self.N):
                y_j = self.y_train[j]
                x_j = self.x_train[j]

                #new_num = y_j * self.alpha_average[j] * self.kernel.get(x_i, x_j)
                new_num = self.alpha_average[j] * self.kernel.get(x_i, x_j)
                the_sum += new_num
            the_sum += self.b_average
            if the_sum > 0:
                result[i] = 1
            elif the_sum < 0:
                result[i] = -1
        return result

    def SGD(self, X, y):
        '''
        描述：用SGD方法求得在hinge loss损失函数下alpha的梯度
        参数：
            X：(N, D)数据矩阵
            Y：(N, 1)数据向量
        返回：
            loss：hinge loss值
            dalpha：alpha的梯度
            db: b的梯度
        '''
        # initialize the gradient and loss as zero
        loss = 0.0
        dalpha = np.zeros(self.alpha.shape)  
        db = 0
        N = X.shape[0]
        D = X.shape[1]

        #随机sample一个minibatch
        indexs = np.random.choice(N, self.batch_size)
        X_batch = X[indexs]
        Y_batch = y[indexs]
        K_batch = self.kernel.K[indexs][:]
        hinge_loss = np.ones(self.batch_size) - Y_batch * np.matmul(K_batch, self.alpha)
        mask = hinge_loss > 0

        mask_y = Y_batch * mask
        new_dalpha = -np.matmul(mask_y, K_batch)
        dalpha += new_dalpha / self.batch_size
        db += -np.mean(mask_y)
        loss += np.sum(hinge_loss * mask) / self.batch_size

        loss += self.reg_weight * np.matmul(self.alpha.T, np.matmul(self.kernel.K, self.alpha)) / 2
        dalpha += self.reg_weight * np.matmul(self.kernel.K, self.alpha.T)

        #正则化
        '''if self.reg_type == 2 :
            dalpha += 2 * self.reg_weight * np.matmul(self.kernel.K, self.alpha.T)
            loss += self.reg_weight * np.matmul(self.alpha.T, np.matmul(self.kernel.K, self.alpha))
            db += 2 * self.reg_weight * self.b
        '''
        return loss, dalpha, db
