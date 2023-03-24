import os
import cv2
import random
from typing import Tuple
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import Dropout
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchmetrics
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
import matplotlib.pyplot as plt

import numpy as np

from compare_snapshots import parse_classes
from experiments import WHEAT_ALL_JUSTIFIED_EXP


class HyperDataset(Dataset):
    def __init__(self, x_data, y_data, y_to_long=False):
        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data)
        self.y_data = self.y_data.long() if y_to_long else self.y_data.float()

        self.x_data = torch.nn.functional.normalize(self.x_data)

        self.len = len(self.x_data)  # Size of data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def get_hp_dataloaders() -> Tuple[DataLoader, DataLoader]:
    classes_features_dict = parse_classes(
        classes_dict={
            **WHEAT_ALL_JUSTIFIED_EXP
        },
        max_files_in_dir=30
    )
    interm_list = [
        (snapshot.name, class_name, snapshot.bands['all'].padded_data.swapaxes(0, 2).swapaxes(1, 2))
        for class_name, class_snapshots in classes_features_dict.items()
        for snapshot in class_snapshots
    ]

    # train-test split
    train_ids = [i for i, (_, class_name, _) in enumerate(interm_list) if class_name in ('health2', 'puccinia phyto2')]
    test_ids = [i for i, (_, class_name, _) in enumerate(interm_list) if class_name in ('health3', 'puccinia phyto3')]
    assert (len(train_ids) + len(test_ids)) == len(interm_list)

    # small wheat leafs filtering
    # saved_interm_list = [tup for tup in interm_list if tup[2].shape[1] < 100 and tup[2].shape[2] < 100]
    saved_interm_list = interm_list

    # small potato leafs filtering
    # saved_interm_list = [tup for tup in interm_list if tup[2].shape[1] < 70 and tup[2].shape[2] < 70]

    print(f"{len(interm_list) - len(saved_interm_list)} snapshots filtered cause of size")

    # data padding
    max_y = max([tup[2].shape[1] for tup in saved_interm_list])
    max_x = max([tup[2].shape[2] for tup in saved_interm_list])

    # channels_num = 23
    all_channels_num = 106
    x_data = np.zeros(shape=(len(saved_interm_list), all_channels_num, max_y, max_x))
    for i, tup in enumerate(saved_interm_list):
        curr_img = tup[2]
        x_data[i, :, :curr_img.shape[1], :curr_img.shape[2]] = curr_img

    # slice range
    # x_data = x_data[:, 10:13]

    # select channels (blue, green, infrared)
    wl_list = [502, 466, 598, 718, 534, 766, 694, 650, 866, 602, 858]
    wl_ids = [(wl - 450) // 4 for wl in wl_list]

    x_data = x_data[:, wl_ids]

    y_data = np.array([0 if 'health' in tup[1] else 1 for tup in saved_interm_list])
    y_data = np.expand_dims(y_data, 1)

    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, shuffle=True, test_size=0.25, random_state=42)
    x_train, y_train = x_data[train_ids], y_data[train_ids]
    x_test, y_test = x_data[test_ids], y_data[test_ids]

    batch_size = 4

    trainloader = torch.utils.data.DataLoader(
        dataset=HyperDataset(x_data=x_train, y_data=y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,

    )

    testloader = torch.utils.data.DataLoader(
        dataset=HyperDataset(x_data=x_test, y_data=y_test),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return trainloader, testloader


class ImgDataset(Dataset):
    def __init__(self, root: str):
        self.img_set = torchvision.datasets.ImageFolder(root=root)

        # TODO possible RAM utilization instead of ImageFolder
        # files = [f"{dir}/{file}" for dir, subdirs, files in os.walk(root) for file in files]
        # self.x_data = np.array([cv2.imread(file) for file in files])
        # self.y_data = np.array([0 if 'dog' in file else 1 for file in files])

    def __getitem__(self, index):
        x_data = np.swapaxes(np.array(self.img_set[index][0].resize((100, 100)), dtype='float32'), 0, 2)
        y_data = np.atleast_1d(self.img_set[index][1])

        x_data = torch.from_numpy(x_data).float()
        y_data = torch.from_numpy(y_data).float()

        return x_data, y_data

    def __len__(self):
        return len(self.img_set)


class ImgDatasetRamUtilizer(Dataset):
    def __init__(self, root: str):
        img_set = torchvision.datasets.ImageFolder(root=root)
        x_data = np.zeros(shape=(len(img_set), 3, 100, 100))
        y_data = np.zeros(shape=(len(img_set), 1))

        for index in range(len(img_set)):
            print(f'loading {index}/{len(img_set)}')
            x_data[index] = np.swapaxes(np.array(img_set[index][0].resize((100, 100)), dtype='float32'), 0, 2)
            y_data[index] = img_set[index][1]

        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).float()

        # normalization
        self.x_data /= 255

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


def get_img_dataloaders() -> Tuple[DataLoader, DataLoader]:
    batch_size = 4

    trainset = ImgDatasetRamUtilizer(root='data/binary_img_ds/train')
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=0
    )

    testset = ImgDatasetRamUtilizer(root='data/binary_img_ds/val')
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size,
        shuffle=True, num_workers=0
    )

    return trainloader, testloader


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, act_func):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool2d(kernel_size, kernel_size)
        self.activation = act_func

    def forward(self, x):
        return self.pool(self.activation(self.conv(x)))


class EpochMetrics:
    def __init__(self, loss: float, acc: float, f1: float, confusion_matrix: np.ndarray):
        self.loss = loss
        self.acc = acc
        self.f1 = f1
        self.confusion_matrix = confusion_matrix

    def __str__(self):
        return f"loss: {self.loss:.3f}, acc: {self.acc:.3f}, f1: {self.f1:.3f}" \
               f"\nconfusion_matrix: \n {self.confusion_matrix}"


class Net(nn.Module):
    def __init__(
            self, channels_num: int, y_size: int, x_size: int,
            lr: float, activation: str, dropout: float,
            verbose=False,
    ):
        print(f"Building cnn for data with shape = [{channels_num}, {y_size}, {x_size}]")
        super().__init__()

        self.verbose = verbose

        if activation == 'relu':
            self.act_func = F.relu
        elif activation == 'tanh':
            self.act_func = F.tanh
        elif activation == 'sigmoid':
            self.act_func = F.sigmoid
        else:
            raise Exception(f"Unexpected activation = {activation}")

        self.conv_layers = nn.ModuleList(
            [
                Dropout(dropout),
                ConvBlock(in_channels=channels_num, out_channels=8, kernel_size=3, act_func=self.act_func),
                Dropout(dropout),
                ConvBlock(in_channels=8, out_channels=16, kernel_size=3, act_func=self.act_func),
                Dropout(dropout),
                ConvBlock(in_channels=16, out_channels=32, kernel_size=3, act_func=self.act_func)
            ]
        )

        for conv_layer in self.conv_layers:
            if isinstance(conv_layer, ConvBlock):
                x_size //= conv_layer.kernel_size
                y_size //= conv_layer.kernel_size

        out_channels = self.conv_layers[-1].out_channels if len(self.conv_layers) != 0 else channels_num

        self.fc_dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(int(x_size * y_size * out_channels), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.acc_metric = torchmetrics.Accuracy(num_classes=2).to(device)
        self.f1_metric = torchmetrics.F1Score(num_classes=2).to(device)
        self.confusion_metric = torchmetrics.ConfusionMatrix(num_classes=2).to(device)

        self.to(device=device)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=lr
        )

        self.train_history = []
        self.test_history = []

    def forward(self, x):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.act_func(self.fc1(self.fc_dropout(x)))
        x = self.act_func(self.fc2(self.fc_dropout(x)))
        x = self.fc3(self.fc_dropout(x))
        return x

    def fit(self):
        assert len(self.train_history) == len(self.test_history)

        epoch_num = 100

        epoch_shift = len(self.train_history)
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            if self.verbose:
                print(f'epoch {epoch_shift + epoch + 1}')

            train_loss_sum = 0.0
            train_acc_sum = 0.0
            train_f1_sum = 0.0
            train_matrix = np.zeros((2, 2))
            for i, data in enumerate(trainloader):
                if self.verbose:
                    print(f'{len(trainloader)}/{i}')

                if i > 500:
                    break
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = transforms(inputs)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item()
                train_acc_sum += self.acc_metric(torch.sigmoid(outputs).int(), labels.int()).item()
                train_f1_sum += self.f1_metric(torch.sigmoid(outputs).int(), labels.int()).item()
                train_matrix += self.confusion_metric(torch.sigmoid(outputs).int(), labels.int()).cpu().detach().numpy()

            self.train_history.append(
                EpochMetrics(
                    loss=train_loss_sum / len(trainloader),
                    acc=train_acc_sum / len(trainloader),
                    f1=train_f1_sum / len(trainloader),
                    confusion_matrix=train_matrix
                )
            )

            test_loss_sum = 0.0
            test_acc_sum = 0.0
            test_f1_sum = 0.0
            test_matrix = np.zeros((2, 2))
            for i, data in enumerate(testloader):
                if i > 125:
                    break
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward + backward + optimize
                with torch.no_grad():
                    inputs = transforms(inputs)
                    outputs = self(inputs)

                    test_loss_sum += self.criterion(outputs, labels).item()
                    test_acc_sum += self.acc_metric(torch.sigmoid(outputs).int(), labels.int()).item()
                    test_f1_sum += self.f1_metric(torch.sigmoid(outputs).int(), labels.int()).item()
                    test_matrix += self.confusion_metric(torch.sigmoid(outputs).int(),
                                                         labels.int()).cpu().detach().numpy()

            self.test_history.append(
                EpochMetrics(
                    loss=test_loss_sum / len(testloader),
                    acc=test_acc_sum / len(testloader),
                    f1=test_f1_sum / len(testloader),
                    confusion_matrix=test_matrix
                )
            )

            if self.verbose:
                print(f"train metrics [{self.train_history[-1]}]")
                print(f"test  metrics [{self.test_history[-1]}]")

        self.draw_history()

    def draw_history(self):
        fig, axes = plt.subplots(nrows=3, ncols=1)

        axes[0].plot([epoch_metric.loss for epoch_metric in self.train_history])
        axes[0].plot([epoch_metric.loss for epoch_metric in self.test_history])
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('loss')

        axes[1].plot([epoch_metric.acc for epoch_metric in self.train_history])
        axes[1].plot([epoch_metric.acc for epoch_metric in self.test_history])
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('acc')

        axes[2].plot([epoch_metric.f1 for epoch_metric in self.train_history])
        axes[2].plot([epoch_metric.f1 for epoch_metric in self.test_history])
        axes[2].set_xlabel('epoch')
        axes[2].set_ylabel('f1')

        plt.legend(['train', 'test'], loc='upper right')
        fig.tight_layout()
        fig.show()


classes = {0: 'health', 1: 'phyto'}


def draw_predicts(ex_num: int = 4):
    fig, axes = plt.subplots(nrows=1, ncols=ex_num)
    for ax in axes:
        batch_image, batch_label = random.choice(testloader.dataset)
        ax.imshow(batch_image.cpu().detach().numpy()[0])
        ax.set_title(classes[int(batch_label)])
    fig.tight_layout()
    fig.show()


def get_random_predict():
    tup = random.choice(testloader.dataset)
    return torch.sigmoid(net(tup[0].unsqueeze(0))), tup[1]


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device} device choose")

    trainloader, testloader = get_img_dataloaders()
    # trainloader, testloader = get_hp_dataloaders()

    transforms = torch.nn.Sequential(
        # transforms.RandomRotation(degrees=90),
        # transforms.RandomVerticalFlip(p=0.5)
        # transforms.RandomErasing()
    )

    print(f"train size = {len(trainloader.dataset)} snapshots")
    print(f"test size = {len(testloader.dataset)} snapshots")
    classes = ('health', 'phyto')

    params_grid = ParameterGrid(
        {'lr': (0.001, 0.0001), 'activation': ['relu', 'tanh'], 'dropout': [0.1, 0.25]}
    )
    res_list = []
    for i, params_dict in enumerate(params_grid):
        print(f'{i + 1}/{len(params_grid)}: fitting scenario ')
        net = Net(*np.array(trainloader.dataset[0][0]).shape, **params_dict, verbose=True)
        net.fit()
        res_list.append(
            {
                **params_dict,
                'loss': min([em.loss for em in net.test_history]),
                'acc': max([em.acc for em in net.test_history]),
                'f1': max([em.f1 for em in net.test_history])
            }
        )

    res_df = pd.DataFrame(res_list)
    get_random_predict()
    draw_predicts(ex_num=4)
