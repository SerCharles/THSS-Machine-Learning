import matplotlib.pyplot as plt
import numpy as np 

def plot_curves(loss_train):
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
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
