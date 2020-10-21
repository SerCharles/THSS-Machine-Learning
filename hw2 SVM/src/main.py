import argparse
from dataloader import load_data
from utils import plot_curves
from classifiers.LinearSVM import LinearSVM
from classifiers.RbfSVM import RbfSVM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose model")
    parser.add_argument("--type", type=str, default="Kernel", help="Linear/Kernel")
    parser.add_argument("--norm", type=int, default=1, help="Normalize or not")

    args = parser.parse_args()
    data = load_data(args.norm)

    if args.type == 'Linear':
        svm = LinearSVM(data)
    else: 
        svm = RbfSVM(data)
    loss_train_list, loss_test, acc_test = svm.run()
    plot_curves(loss_train_list)