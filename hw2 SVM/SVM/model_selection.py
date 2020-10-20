from classifiers.LinearSVM import LinearSVM
from classifiers.RbfSVM import RbfSVM
from dataloader import load_data
from utils import plot_curves
import argparse

def grid_search(args):
    '''
    描述：用grid_search找到最优超参数
    参数：args
    返回：最优的loss_train_list, loss_eval_list, acc_train_list, acc_eval_list, loss_test, acc_test
    '''
    model_type = args.type
    if model_type == 'Linear':
        reg_choices = [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
        reg_type_choices = [1, 2]
        lr_choices = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        bs_choices = [1, 16, 32, 64, 128, 256]
    else: 
        reg_choices = [0, 1e-2, 1e-1, 1, 10]
        reg_type_choices = [2]
        lr_choices = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        bs_choices = [1, 16, 32, 64, 128, 256]
    best_reg = 0
    best_reg_type = 0
    best_lr = 0
    best_bs = 0
    best_acc = 0
    best_loss_train_list = []
    best_loss_eval_list = []
    best_acc_train_list = []
    best_acc_eval_list = []
    best_loss_test = 0
    best_acc_test = 0
    print("Model Type:", model_type)
    for reg in reg_choices:
        for reg_type in reg_type_choices:
            for lr in lr_choices:
                for bs in bs_choices:
                    data = load_data(args.norm)
                    if model_type == 'Linear':
                        
                        the_model = LinearSVM(data, batch_size = bs, learning_rate = lr, epochs = 100, reg_type = reg_type, reg_weight= reg, whether_print=False)
                        loss_train_list, loss_eval_list, acc_train_list, acc_eval_list, loss_test, acc_test = the_model.run()
                    elif model_type == 'Kernel':
                        the_model = RbfSVM(data, batch_size = bs, learning_rate = lr, epochs = 100, reg_type = reg_type, reg_weight= reg, whether_print=False)
                        loss_train_list, loss_eval_list, acc_train_list, acc_eval_list, loss_test, acc_test = the_model.run() 
                    the_acc = max(acc_eval_list)   
                    print("Current acc", the_acc, "Current reg_type:", reg_type, "Current reg weight:", reg, "Current lr:", lr, "Current bs:", bs)
                    if((the_acc + acc_test) / 2 > best_acc and the_acc < 1):
                        best_acc = (the_acc + acc_test) / 2
                        best_reg = reg
                        best_reg_type = reg_type
                        best_lr = lr
                        best_bs = bs
                        best_loss_train_list = loss_train_list
                        best_acc_train_list = acc_train_list
                        best_loss_eval_list = loss_eval_list
                        best_acc_eval_list = acc_eval_list
                        best_loss_test = loss_test
                        best_acc_test = acc_test
    print("Best learning rate:{}\nBest batch size:{}\nBest reg type:{}\nBest reg weight:{}\nBest test accuracy:{:.4f}\nBest accuracy:{:.4f}".format(best_lr, best_bs, best_reg_type, best_reg, best_acc_test, best_acc))
    return best_loss_train_list, best_loss_eval_list, best_acc_train_list, best_acc_eval_list, best_loss_test, best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose model")
    parser.add_argument("--type", type=str, default="Linear", help="Linear/Kernel")
    parser.add_argument("--norm", type=int, default=1, help="Normalize or not")
    args = parser.parse_args()
    best_loss_train_list, best_loss_eval_list, best_acc_train_list, best_acc_eval_list, best_loss_test, best_acc_test = grid_search(args)
