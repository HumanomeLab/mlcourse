#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import os
import copy
import argparse
import time
import numpy as np
import pandas as pd

# オプションの設定
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
#parser.add_argument('object', metavar='OBJ',
#                    help='path to object positions')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
#parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--device', default="cpu", type=str,
                    help='GPU id to use. like cuda or cuda:3. If you use CPU, type "cpu"')
parser.add_argument('--outdir', default=".", type=str,
                    help='Result output directory')


SEQ_LENGTH = 100


def make_dataset(datadir):
    pos_seq = "SRX356455.05_peak_seq_100.txt"
    neg_seq = "SRX356455.05_random_seq_100.txt"
    # id      chr     start   end     seq
    data = pd.read_csv(os.path.join(datadir, "sequences", pos_seq), sep="\t")
    sequences = []
    classes = []
    for index, row in data[["id", "seq"]].iterrows():
        y = 1
        seq_vector = seq2vector(row["seq"])
        if len(seq_vector) == 0:
            continue
        sequences.append(seq2vector(row["seq"]))
        classes.append(np.array(y))
    data = pd.read_csv(os.path.join(datadir, "sequences", neg_seq), sep="\t")
    for index, row in data[["id", "seq"]].iterrows():
        y = 0
        seq_vector = seq2vector(row["seq"])
        if len(seq_vector) == 0:
            continue
        sequences.append(seq2vector(row["seq"]))
        classes.append(np.array(y))
    return sequences, classes


def seq2vector(seq):
    if type(seq) is not str: # Case on Null sequence
        return np.zeros((0,0))
    seq_array = np.zeros((4, SEQ_LENGTH))
    flag = 0
    for i in range(SEQ_LENGTH):
        s = seq[i]
        if s == "a" or s == "A":
            seq_array[0, i] = 1
        elif s == "c" or s == "C":
            seq_array[1, i] = 1
        elif s == "g" or s == "G":
            seq_array[2, i] = 1
        elif s == "t" or s == "T":
            seq_array[3, i] = 1
        else:
            flag += 1
    if len(seq) == flag: # Case on N sequence
        return np.zeros((0,0))
    seq_array = seq_array.astype(np.float32)
    return seq_array


class DatasetFolder(data.Dataset):
    def __init__(self, X, y):
        self.samples = X
        self.targets = y
        self.transforms = transforms.Compose([
            ToTensorOfTarget()
        ])

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = self.transforms(sample)
        target = self.targets[index]
        target = self.transforms(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


class ToTensorOfTarget(object):
    def __call__(self, target):
        return torch.from_numpy(target)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(4, 8, 3, padding=1) # 100
        self.pool = nn.MaxPool1d(2) # 50
        self.conv2 = nn.Conv1d(8, 16, 3, padding=1) # 50
        self.fc1 = nn.Linear(16 * 25 * 1, 20) # channel_num * x * y
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 100 -> 100 -> 50
        x = self.pool(F.relu(self.conv2(x))) # 50 -> 50 -> 25
        x = x.view(-1, 16 * 25 * 1) # channel_num * x * y
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def run_cnn(datadir, device_name="cpu", batch_size=64, workers=0, lr=0.01, momentum=0.9, epochs=10, outdir="."):
    global dataloaders, device, dataset_sizes
    # CPUで実行するか、GPUで実行するかの指定
    device = torch.device(device_name)
    print("=> using device: %s" % device)

    # 全体を、training, valid, testに分ける。ここでは、3:1:1 に分割。
    # training + valid が、機械学習の training data 相当。
    X, y = make_dataset(datadir)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size = 0.20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size = 0.25
    )


    sequence_datasets = {
        'train':DatasetFolder(X_train, y_train),
        'val':DatasetFolder(X_val, y_val),
        'test': DatasetFolder(X_test, y_test)
    }

    # バッチサイズ分のデータを読み込む。
    # training はデータをシャッフルし、読み込み始める画像をランダムにする。
    # 他はシャッフルの必要なし。
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            sequence_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers),
        'val': torch.utils.data.DataLoader(
            sequence_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers),
        'test': torch.utils.data.DataLoader(
            sequence_datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers)
    }
    dataset_sizes = {x: len(sequence_datasets[x]) for x in ['train', 'val', 'test']}

    # シンプルなCNNモデルを作る
    print("=> CNN")
    model = Net()
    model = model.to(device)

    # Loss関数の定義。
    # Regression なので、CrossEntropy から、MSELossに変更
    criterion = nn.CrossEntropyLoss()
    # optimizer の定義
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # 10 エポックごとに学習率を0.1倍する
    # 値は、ここでは固定してしまっているが、本来は可変。
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    # 実際の学習を実施する
    # 結果出力用ファイルのprefix
    outpath = os.path.join(outdir, "cnn_feature_b%d_lr%f_m%f_e%d" % (batch_size, lr, momentum, epochs))
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, outpath, num_epochs=epochs)
    # 学習が終わったら、結果を保存する。
    torch.save(model.state_dict(), 'model.pkl')
    # テストデータでの精度を求める
    print_test_accuracy(model, criterion, optimizer, 'test')


def print_test_accuracy(model, criterion, optimizer, phase):
    running_loss = 0.0
    running_corrects = 0
    model.train(False)

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        #optimizer.zero_grad()

        # 訓練のときだけ履歴を保持する
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            #_, classnums = torch.max(labels, 1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # 統計情報
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    # サンプル数で割って平均を求める
    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]
    print('On Test:\tLoss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


def train_model(model, criterion, optimizer, scheduler, outpath, num_epochs=25):
    since = time.time()
    # 途中経過でモデル保存するための初期化
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # 時間計測用
    end = time.time()

    print(model)
    print()

    for epoch in range(num_epochs):
        print('Epoch:{}/{}'.format(epoch, num_epochs - 1), end="")

        # 各エポックで訓練+バリデーションを実行
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # training mode
            else:
                model.train(False)  # evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                #labels = labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 訓練のときだけ履歴を保持する
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_, classnums = torch.max(labels, 1)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 統計情報
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            # サンプル数で割って平均を求める
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('\t{} Loss: {:.4f} Acc: {:.4f} Time: {:.4f}'.format(phase, epoch_loss, epoch_acc, time.time()-end), end="")

            # 精度が改善したらモデルを保存する
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            end = time.time()

        print()

    time_elapsed = time.time() - since
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    global args
    args = parser.parse_args()
    device_name = args.device
    data = args.data
    batch_size = args.batch_size
    workers = args.workers
    lr = args.lr
    momentum = args.momentum
    epochs = args.epochs
    outdir = args.outdir
    run_cnn(data, device_name, batch_size, workers, lr, momentum, epochs, outdir)


if __name__ == '__main__':
    main()

