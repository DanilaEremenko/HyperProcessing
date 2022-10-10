import random
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import torch.optim as optim
import torchmetrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np

from compare_snapshots import parse_classes


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classes_features_dict = parse_classes(
        classes_dict={
            'health': [
                'csv/control/gala-control-bp-1_000',
                'csv/control/gala-control-bp-2_000',
                'csv/control/gala-control-bp-3_000',
                'csv/control/gala-control-bp-4_000',
            ],
            'phyto1': [
                'csv/phytophthora/gala-phytophthora-bp-1_000',
                'csv/phytophthora/gala-phytophthora-bp-5-1_000',
                # 'csv/phytophthora-ps-2_2-5_2/gala-phytophthora-2_2-5_2-1_000'
            ],
            'phyto2': [
                'csv/phytophthora/gala-phytophthora-bp-2_000',
                'csv/phytophthora/gala-phytophthora-bp-6-2_000',
                # 'csv/phytophthora-ps-2_2-5_2/gala-phytophthora-2_2-5_2-2_000'
            ],
            # 'phyto3': [
            #     'csv/phytophthora/gala-phytophthora-bp-3_000',
            #     'csv/phytophthora/gala-phytophthora-bp-7-3_000',
            # ],
            # 'phyto4': [
            #     'csv/phytophthora/gala-phytophthora-bp-4_000',
            #     'csv/phytophthora/gala-phytophthora-bp-8-4_000',
            # ]

        },
        max_files_in_dir=30
    )
    interm_list = [
        (snapshot.name, class_name, snapshot.bands['infrared'].padded_data.swapaxes(0, 2).swapaxes(1, 2))
        for class_name, class_snapshots in classes_features_dict.items()
        for snapshot in class_snapshots
    ]

    filt_interm_list = [tup for tup in interm_list if tup[2].shape[1] < 70 and tup[2].shape[2] < 70]

    print(f"{len(interm_list) - len(filt_interm_list)} snapshots filtered cause of size")

    # data padding
    max_y = max([tup[2].shape[1] for tup in filt_interm_list])
    max_x = max([tup[2].shape[2] for tup in filt_interm_list])

    channels_num = 23
    x_data = np.zeros(shape=(len(filt_interm_list), channels_num, max_y, max_x))
    for i, tup in enumerate(filt_interm_list):
        curr_img = tup[2]
        x_data[i, :, :curr_img.shape[1], :curr_img.shape[2]] = curr_img

    # slice range
    x_data = x_data[:, 10:13]

    y_data = np.array([0 if 'health' in tup[1] else 1 for tup in filt_interm_list])
    y_data = np.expand_dims(y_data, 1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, shuffle=True, test_size=0.25, random_state=42)

    return x_train, y_train, x_test, y_test


class HyperDataset(Dataset):
    def __init__(self, x_data, y_data, device='cpu', y_to_long=False):
        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data)
        self.y_data = self.y_data.long() if y_to_long else self.y_data.float()

        self.x_data = torch.nn.functional.normalize(self.x_data)

        self.len = len(self.x_data)  # Size of data
        self.device = device

    def __getitem__(self, index):
        return self.x_data[index].to(self.device), self.y_data[index].to(self.device)

    def __len__(self):
        return self.len


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool2d(kernel_size, kernel_size)

    def forward(self, x):
        return self.pool(F.relu(self.conv(x)))


class Net(nn.Module):
    def __init__(self, channels_num: int, y_size: int, x_size: int):
        print(f"Building cnn for data with shape = [{channels_num}, {y_size}, {x_size}]")
        super().__init__()

        self.conv_layers = nn.ModuleList(
            [
                ConvBlock(in_channels=3, out_channels=8, kernel_size=3),
                ConvBlock(in_channels=8, out_channels=16, kernel_size=3),
                # ConvBlock(in_channels=16, out_channels=32, kernel_size=3)
            ]
        )

        for conv_layer in self.conv_layers:
            x_size //= conv_layer.kernel_size
            y_size //= conv_layer.kernel_size

        out_channels = self.conv_layers[-1].out_channels if len(self.conv_layers) != 0 else channels_num

        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(int(x_size * y_size * out_channels), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(self.dropout(x)))
        x = torch.relu(self.fc2(self.dropout(x)))
        x = self.fc3(self.dropout(x))
        return x


class EpochMetrics:
    def __init__(self, loss: float, acc: float, f1: float, confusion_matrix: np.ndarray):
        self.loss = loss
        self.acc = acc
        self.f1 = f1
        self.confusion_matrix = confusion_matrix

    def __str__(self):
        return f"loss: {self.loss:.3f}, acc: {self.acc:.3f}, f1: {self.f1:.3f}" \
               f"\nconfusion_matrix: \n {self.confusion_matrix}"


def draw_history():
    fig, axes = plt.subplots(nrows=3, ncols=1)

    axes[0].plot([epoch_metric.loss for epoch_metric in train_history])
    axes[0].plot([epoch_metric.loss for epoch_metric in test_history])

    axes[1].plot([epoch_metric.acc for epoch_metric in train_history])
    axes[1].plot([epoch_metric.acc for epoch_metric in test_history])

    axes[2].plot([epoch_metric.f1 for epoch_metric in train_history])
    axes[2].plot([epoch_metric.f1 for epoch_metric in test_history])

    fig.show()


classes = {0: 'health', 1: 'phyto'}


def draw_predicts(ex_num: int = 4):
    fig, axes = plt.subplots(nrows=ex_num, ncols=1)
    for ax in axes:
        batch_image, batch_label = random.choice(testloader.dataset)
        ax.imshow(batch_image.cpu().detach().numpy()[0])
        ax.set_title(classes[int(batch_label)])
    fig.tight_layout()
    fig.show()


def get_random_predict():
    tup = random.choice(testloader.dataset)
    return torch.sigmoid(net(tup[0].unsqueeze(0))), tup[1]


def fit(epoch_num: int, train_history: List[EpochMetrics], test_history: List[EpochMetrics]):
    assert len(train_history) == len(test_history)

    epoch_shift = len(train_history)
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        print(f'epoch {epoch_shift + epoch + 1}')

        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_f1_sum = 0.0
        train_matrix = np.zeros((2, 2))
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = transforms(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += acc_metric(torch.sigmoid(outputs).int(), labels.int()).item()
            train_f1_sum += f1_metric(torch.sigmoid(outputs).int(), labels.int()).item()
            train_matrix += confusion_metric(torch.sigmoid(outputs).int(), labels.int()).cpu().detach().numpy()

        train_history.append(
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
            inputs, labels = data

            # forward + backward + optimize
            with torch.no_grad():
                inputs = transforms(inputs)
                outputs = net(inputs)

                test_loss_sum += criterion(outputs, labels).item()
                test_acc_sum += acc_metric(torch.sigmoid(outputs).int(), labels.int()).item()
                test_f1_sum += f1_metric(torch.sigmoid(outputs).int(), labels.int()).item()
                test_matrix += confusion_metric(torch.sigmoid(outputs).int(), labels.int()).cpu().detach().numpy()

        test_history.append(
            EpochMetrics(
                loss=test_loss_sum / len(testloader),
                acc=test_acc_sum / len(testloader),
                f1=test_f1_sum / len(testloader),
                confusion_matrix=test_matrix
            )
        )

        print(f"train metrics [{train_history[-1]}]")
        print(f"test  metrics [{test_history[-1]}]")

    draw_history()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device} device choose")

    x_data_train, y_data_train, x_data_test, y_data_test = get_data()

    batch_size = 4

    transforms = torch.nn.Sequential(
        # transforms.RandomRotation(degrees=90),
        # transforms.RandomVerticalFlip(p=0.5)
        # transforms.RandomErasing()
    )

    trainloader = torch.utils.data.DataLoader(
        dataset=HyperDataset(x_data=x_data_train, y_data=y_data_train, device=device),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,

    )

    testloader = torch.utils.data.DataLoader(
        dataset=HyperDataset(x_data=x_data_test, y_data=y_data_test, device=device),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    print(f"train size = {len(trainloader.dataset)} snapshots")
    print(f"test size = {len(testloader.dataset)} snapshots")
    classes = ('health', 'phyto')

    net = Net(*x_data_train.shape[1:])
    net.to(device=device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    acc_metric = torchmetrics.Accuracy(num_classes=2).to(device)
    f1_metric = torchmetrics.F1Score(num_classes=2).to(device)
    confusion_metric = torchmetrics.ConfusionMatrix(num_classes=2).to(device)

    optimizer = optim.Adam(
        net.parameters(),
        lr=0.001
    )

    train_history = []
    test_history = []

    fit(epoch_num=10, train_history=train_history, test_history=test_history)

    get_random_predict()
    draw_predicts(ex_num=4)
