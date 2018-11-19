#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
import copy
import argparse
import time
import numpy as np
import pandas as pd

# オプションの設定
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
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


# 画像の幅と高さ
WIDTH = 224
HEIGHT = 224


def make_dataset(dir):
    images = []
    labels = []
    dir = os.path.expanduser(dir)
    data = pd.read_csv("data/yeast_his3.csv")
    for _, row in data[["Cgroup","PhotoID", "CellID"]].iterrows():
        filename = "C_yor202w_0_0_%d_%d" % (row["PhotoID"], row["CellID"])
        image_path = os.path.join(dir, filename + ".png")
        y = [0, 0, 0, 0]
        if row["Cgroup"] == "no":
            y = [1, 0, 0, 0]
        elif row["Cgroup"] == "small":
            y = [0, 1, 0, 0]
        elif row["Cgroup"] == "medium":
            y = [0, 0, 1, 0]
        elif row["Cgroup"] == "large":
            y = [0, 0, 0, 1]
        elif row["Cgroup"] == "complex":
            y = [0, 0, 0, 0]
        images.append(image_path)
        labels.append(np.array(y))
    return images, labels


class DatasetFolder(data.Dataset):
    def __init__(self, X, y, loader, transform=None, target_transform=None):
        self.loader = loader
        self.samples = X
        self.targets = y
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


#class YeastImageDataset(DatasetFolder):
#    def __init__(self, X, y, transform=None, target_transform=None,
#                 loader=pil_loader):
#        super(YeastImageDataset, self).__init__(X, y, pil_loader,
#                                          transform=transform,
#                                          target_transform=target_transform)
#        self.imgs = self.samples


class ToTensorOfTarget(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, target):
        return torch.from_numpy(target)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=(1,1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=(1,1))
        self.fc1 = nn.Linear(16 * 56 * 56, 120) # channel_num * x * y
        self.fc2 = nn.Linear(120, 4)

    def forward(self, x):
        # ** convolution layers **
        # 224 x 224 -> 112 x 112 
        x = self.pool(F.relu(self.conv1(x)))
        # 112 x 112 -> 56 x 56
        x = self.pool(F.relu(self.conv2(x)))
        # convert to 1-dim
        x = x.view(-1, 16 * 56 * 56) # channel_num * x * y
        # ** classification layers **
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def run_cnn(data, device_name="cpu", batch_size=64, workers=0, lr=0.01, momentum=0.9, epochs=10, outdir="."):
    # train_model 関数でも利用する一部の変数をグローバル化
    global dataloaders, device, dataset_sizes, model_str
    # CPUで実行するか、GPUで実行するかの指定
    device = torch.device(device_name)
    print("=> using device: %s" % device)

    # 全体を、training, valid, testに分ける。ここでは、3:1:1 に分割。
    # training + valid が、機械学習の training data 相当。
    datadir = os.path.join(data, 'images')
    X, y = make_dataset(datadir)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size = 0.20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size = 0.25
    )


    # 画像の輝度値を補正するための関数を設定。
    # ResNet等のPre-trained model 学習時に利用されていた値
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # training (or validation）時に、画像に対して変換を加える場合は、
    # ここに記述する。ResizeやFlipなど。
    # 参照：https://pytorch.org/docs/stable/torchvision/transforms.html
    # 変換のあと、pytorchで扱うために、Tensor型に変換してあげる必要あり。
    # normalizeは、Tensor型に変換したあと、実施
    data_transforms = {
        # training data用。必要ならFlipやリサイズを行う
        # このサンプルでは、特段の加工は行わない。
        'train': transforms.Compose([
            transforms.Resize((WIDTH, HEIGHT)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            normalize
        ]),
        # validation用。通常はFlip等は行わない。
        'val': transforms.Compose([
            transforms.Resize((WIDTH, HEIGHT)),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize((WIDTH, HEIGHT)),
            transforms.ToTensor(),
            normalize
        ])
    }

    target_transforms = transforms.Compose([
            ToTensorOfTarget()
    ])


    # 画像とクラスの読み込み用の関数を定義
    image_datasets = {
        'train':DatasetFolder(X_train, y_train, pil_loader,
                                  data_transforms['train'],
                                  target_transforms),
        'val':DatasetFolder(X_val, y_val, pil_loader,
                                 data_transforms['val'],
                                 target_transforms),
        'test': DatasetFolder(X_test, y_test, pil_loader,
                                 data_transforms['test'],
                                 target_transforms)
    }

    # バッチサイズ分のデータを読み込む。
    # training はデータをシャッフルし、読み込み始める画像をランダムにする。
    # 他はシャッフルの必要なし。
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

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
    outpath = os.path.join(outdir, "cnn_b%d_lr%f_m%f_e%d" % (batch_size, lr, momentum, epochs))
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
        labels = labels.float()
        inputs = inputs.to(device)
        labels = labels.to(device)

        #optimizer.zero_grad()

        # 訓練のときだけ履歴を保持する
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, classnums = torch.max(labels, 1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, classnums)

        # 統計情報
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == classnums)

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
                labels = labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 訓練のときだけ履歴を保持する
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, classnums = torch.max(labels, 1)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, classnums)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 統計情報
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == classnums)

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
