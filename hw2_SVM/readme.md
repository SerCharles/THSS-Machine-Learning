请用python3在src目录下运行（我是python 3.7.1）

**运行主函数**

```shell
python main.py --type=Linear
python main.py --type=Kernel
```

**运行参数选择函数**

```shell
python model_selection.py --type=Linear
python model_selection.py --type=Kernel
```

**运行sklearn**

```shell
python main_sklearn.py --type=linear
python main_sklearn.py --type=rbf
```

我的数据默认归一化过，要取消请后加 --norm=0