from sklearn.svm import SVC
from dataloader import load_data
import argparse

def run_sklearn(args):
    model_type = args.type
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.norm)
    svc = SVC(kernel = model_type)
    svc.fit(X_train, y_train)
    y_result = svc.predict(X_test)
    result = 0
    for i in range(len(y_result)):
        if y_result[i] == y_test[i]:
            result += 1
    sklearn_acc = result / len(y_result)
    print('Sklearn test accuracy of model type {} is {:.4f}'.format(model_type, sklearn_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose model")
    parser.add_argument("--type", type=str, default="linear", help = "linear/rbf")
    parser.add_argument("--norm", type=int, default=1, help="Normalize or not")
    args = parser.parse_args()
    run_sklearn(args.type)