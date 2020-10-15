import matplotlib.pyplot as plt
import numpy as np 

def plot_curves(loss_train, loss_eval, acc_train, acc_eval):
    '''
    描述：绘制训练-测试曲线
    参数：训练，测试的准确度
    返回：无
    '''
    x_axix = []
    for i in range(len(loss_train)):
        x_axix.append(i)
    plt.title('Loss Comparison')
    plt.plot(x_axix, loss_train, color='red', label='training loss')
    plt.plot(x_axix, loss_eval, color='blue', label='valid loss')
    plt.legend() # 显示图例

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.title('Accuracy Comparison')
    plt.plot(x_axix, acc_train, color='red', label='training accuracy')
    plt.plot(x_axix, acc_eval, color='blue', label='valid accuracy')
    plt.legend() # 显示图例

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()