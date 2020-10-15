import argparse
from dataloader import load_data
from utils import plot_curves
from classifiers.LinearSVM import LinearSVM
from classifiers.RbfSVM import RbfSVM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose model")
    parser.add_argument("--type", type=str, default="Linear", help="Linear/Kernel")
    args = parser.parse_args()
    data = load_data()

    if args.type == 'Linear':
        svm = LinearSVM(data)
    else: 
        svm = RbfSVM(data)
    loss_train_list, loss_eval_list, acc_train_list, acc_eval_list, loss_test, acc_test = svm.run()
    plot_curves(loss_train_list, loss_eval_list, acc_train_list, acc_eval_list)