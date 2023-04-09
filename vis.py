import matplotlib.pyplot as plt
import os
import numpy as np


def vis_loss(log_path='Loss_Curve.txt'):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    train_loss_list = list()
    test_loss_list = list()
    test_acc_list = list()
    for line in lines:
        train_loss, test_loss, test_acc = line.split()
        train_loss_list.append(float(train_loss))
        test_loss_list.append(float(test_loss))
        test_acc_list.append(float(test_acc))

    x = range(len(train_loss_list))

    plt.plot(x, train_loss_list, ms=10, label='train_loss')
    plt.plot(x, test_loss_list, ms=10, label='test_loss')
    plt.plot(x, test_acc_list, ms=10, label='test_acc')
    plt.xlabel('iteration')
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('Loss.jpg', dpi=900)
    plt.clf()


def vis_param(file_path='checkpoint/'):
    for f in os.listdir(file_path):
        if 'npy' in f:
            data = np.load(file_path + str(f))
            name = f.split('.')[0]
            data = data.reshape(-1)
            plt.hist(data, bins=50)
            plt.xlabel('Value')
            plt.ylabel("Freq")
            plt.savefig(name + '.jpg', dpi=900)
            plt.clf()


if __name__ == '__main__':
    # vis_param()
    vis_loss()
