import numpy as np
import argparse
import os
import gzip
import struct


def load_mnist_train(path='dataset', mode='train'):
    # 读取文件
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% mode)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% mode)
    #使用gzip打开文件
    with gzip.open(labels_path, 'rb') as lbpath:
        #使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
        #这样读到的前两个数据分别是magic number和样本个数
        magic, n = struct.unpack('>II',lbpath.read(8))
        #使用np.fromstring读取剩下的数据，lbpath.read()表示读取所有的数据
        labels = np.fromstring(lbpath.read(),dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromstring(imgpath.read(),dtype=np.uint8).reshape(len(labels), 784)
    label_matrix = np.zeros((images.shape[0], 10))
    length = len(labels)
    for i in range(length):
        label_matrix[i][labels[i]] = 1
    return images, label_matrix


def create_hyperparam(x, y, hidden=512):
    n_x = x.shape[1]
    n_y = y.shape[1]
    n_h = hidden
    return n_x, n_y, n_h


def init_param(n_x, n_y, n_h):
    w1 = np.random.randn(n_x, n_h) * 0.01
    b1 = np.random.randn(n_h)
    w2 = np.random.randn(n_h, n_y) * 0.01
    b2 = np.random.randn(n_y)
    return w1, w2, b1, b2


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x) + 1e-6)


def forward(w1, b1, w2, b2, x):
    y1 = np.dot(x, w1) + b1
    y1_act = tanh(y1)
    y2 = np.dot(y1_act, w2) + b2
    return sigmoid(y2), y1_act


def backward(x, y, yhat, a1, w1, w2, b1, b2, N, regular):
    dz2 = (-1/N) * (y*(1-yhat) - yhat*(1-y))
    dw2 = np.dot(a1.T, dz2) + regular * w2
    db2 = dz2.sum(axis=0, keepdims=False) + regular * b2

    dz1 = np.dot(dz2, w2.T) * (1-a1**2)
    dw1 = np.dot(x.T, dz1) + regular * w1
    db1 = dz1.sum(axis=0, keepdims=False) + regular * b1
    grids = {'dw2': dw2, 'db2': db2, 'dw1': dw1, 'db1': db1}
    return grids


def SGD(w1, w2, b1, b2, lr, grids):
    w1 = w1 - lr * grids['dw1']
    w2 = w2 - lr * grids['dw2']
    b1 = b1 - lr * grids['db1']
    b2 = b2 - lr * grids['db2']
    return w1, w2, b1, b2


# def ce_loss(gt, pred, N):
#     assert gt.shape == pred.shape
#     loss = (-1/N)*(np.dot(gt, np.log(pred.T)) + np.dot((1-gt), np.log((1-pred).T)))
#     return loss


def ce_loss(pred, gt):
    y_shift = pred - np.max(pred, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-gt * np.log(y_probability), axis=-1))  # 损失函数
    return loss


def cal_acc(gt, pred):
    acc = np.mean(np.equal(np.argmax(pred, axis=-1),
                                np.argmax(gt, axis=-1)))
    # pred = pred.T
    # for i in range(len(pred)):
    #     if pred[i] < 0.5:
    #         pred[i] = 0
    #     else:
    #         pred[i] = 1
    # acc = (np.dot(gt, pred) + np.dot(1-gt, 1-pred)) / float(len(pred))
    return acc


def train_model(x, y, lr, regular, hidden, num_epoch, batch_size, test_x, test_y):
    n_x, n_y, n_hidden = create_hyperparam(x, y, hidden=hidden)
    w1, w2, b1, b2 = init_param(n_x, n_y, n_hidden)
    n_iters = x.shape[0] // batch_size
    for epoch_idx in range(num_epoch):
        for i in range(n_iters):
            batch_x, batch_y = random_batch(x, y, batch_size)
            pred, a1 = forward(w1, b1, w2, b2, batch_x)

            if i % 50 == 0:
                loss = ce_loss(batch_y, pred)
                acc = cal_acc(batch_y, pred)
                print('epoch: %d, iter: %d, loss: %f, acc: %f' % (epoch_idx, i, loss, acc))
                test_pred, _ = forward(w1, b1, w2, b2, test_x)
                test_loss = ce_loss(test_y, test_pred)
                test_acc = cal_acc(test_y, test_pred)

                line = str(loss) + ' ' + str(test_loss) + ' ' + str(test_acc) + '\r'
                with open('Loss_Curve.txt', 'a+') as f:
                    f.write(line)

            grids = backward(batch_x, batch_y, pred, a1, w1, w2, b1, b2, batch_x.shape[1], regular)
            w1, w2, b1, b2 = SGD(w1, w2, b1, b2, lr, grids)
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return params


def eval_model(params, x, y):
    w1, b1, w2, b2 = params['w1'], params['b1'], params['w2'], params['b2']
    pred, _ = forward(w1, b1, w2, b2, x)
    acc = cal_acc(y, pred)
    print('acc: ', acc)
    return acc


def save_model(params, path):
    np.save(path+'_w1.npy', params['w1'])
    np.save(path+'_w2.npy', params['w2'])
    np.save(path+'_b1.npy', params['b1'])
    np.save(path+'_b2.npy', params['b2'])


def random_batch(data, label, batch_size):
    train_num = data.shape[0]
    search_index = np.random.choice(train_num, batch_size)
    return data[search_index], label[search_index]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='dataset', help="root path of dataset")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512, help="hidden_size")
    parser.add_argument("--lr", type=int, default=0.001, help="learning_rate")
    parser.add_argument("--regular", type=int, default=0.01, help="regular")

    args = parser.parse_args()

    x, y = load_mnist_train(path=args.data_root, mode='train')
    test_x, test_y = load_mnist_train(path=args.data_root, mode='test')

    params = train_model(x, y, args.lr, args.regular, args.hidden_dim, args.num_epoch, args.batch_size, test_x, test_y)
    save_model(params, path='checkpoint')

    eval_model(params, test_x, test_y)
