import numpy as np
import random
from dataloader import load_data

class SVMClassifier(object):
    def __init__(self, batch_size = 200, learning_rate = 1e-3, epochs = 100, reg_type = 0, reg_weight = 1e-5, hinge_weight = 1):
        #初始化参数
        self.W = np.random.randn(8, ) * 0.0001
        self.b = random.random() * 0.0001

        #定义各种超参数
        self.batch_size = batch_size #SGD的minibatch size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_type = 0 #0：不正则化 1:L1正则化 2：L2正则化
        self.reg_weight = reg_weight #正则化的weight
        self.hinge_weight = hinge_weight #hinge loss的权重

    def run(self):
        '''
        描述：主函数
        参数：无
        返回：最优的参数W，b
        '''
        x_train, x_test, y_train, y_test = load_data()
        max_acc = 0
        best_W = self.W.copy()
        best_b = self.b
        for i in range(self.epochs):
            loss_train, acc_train = self.train_or_eval("Train", x_train, y_train, i + 1)
            loss_eval, acc_eval = self.train_or_eval("Eval", x_test, y_test, i + 1)
            if acc_eval > max_acc:
                max_acc = acc_eval
                best_W = self.W.copy()
                best_b = self.b
        return best_W, best_b

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
            if(y_i == 0):
                y_i = -1
            result = y_i * (np.matmul(self.W.T, x_i) + self.b)
            if result >= 0:
                total_right += 1
        acc = total_right / N
        loss, dW, db = self.SGD(X, y)
        if mode == 'Train':
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
        print("{} Epoch:[{}/{}] Accuracy:{:.2f} Loss:{:.2f}".format(mode, epoch, self.epochs, acc, loss))
        return loss, acc

    def SGD(self, X, y):
        '''
        描述：用SGD方法求得在hinge loss损失函数下W，b的梯度
        参数：
            X：(N, D)数据矩阵
            Y：(N, 1)数据向量
        返回：
            loss：hinge loss值
            dW：W的梯度
            db：b的梯度
        '''
        # initialize the gradient and loss as zero
        loss = 0.0
        dW = np.zeros(self.W.shape)  
        db = 0.0
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

            difference = 1 - y_i * (np.matmul(self.W.T, x_i) + self.b)
            if difference > 0:
                dW -= y_i * x_i
                db -= y_i
                loss += difference
        dW = dW / self.batch_size * self.hinge_weight
        db = db / self.batch_size * self.hinge_weight
        loss = loss / self.batch_size * self.hinge_weight

        #考虑||W||
        loss += np.matmul(self.W.T, self.W) / 2
        dW += self.W

        #正则化
        if self.reg_type == 2 :
            dW += 2 * self.reg_weight * self.W
            db += 2 * self.reg_weight * self.b 
        elif self.reg_type == 1: 
            for i in range(D):
                if self.W[i] > 0:
                    dW[i] += self.reg_weight
                elif self.W[i] < 0:
                    dW[i] -= self.reg_weight
            if self.b > 0:
                db += self.reg_weight
            elif self.b < 0:
                db -= self.reg_weight
        
        return loss, dW, db

if __name__ == "__main__":
    svm = SVMClassifier()
    svm.run()