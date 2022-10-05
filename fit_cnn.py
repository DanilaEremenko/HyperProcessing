from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
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
        max_files_in_dir=1
    )
    interm_list = [
        (snapshot.name, class_name, snapshot.bands['infrared'].padded_data.swapaxes(0, 2).swapaxes(1, 2))
        for class_name, class_snapshots in classes_features_dict.items()
        for snapshot in class_snapshots
    ]

    max_y = max([tup[2].shape[1] for tup in interm_list])
    max_x = max([tup[2].shape[2] for tup in interm_list])

    x_data = np.zeros(shape=(len(interm_list), 23, max_y, max_x), dtype='uint8')

    for i, tup in enumerate(interm_list):
        x_data[i] = np.resize(tup[2], (23, max_y, max_x))

    x_data = x_data[:, 10:13]

    y_data = np.array([0 if 'health' in tup[1] else 1 for tup in interm_list])
    y_data = np.expand_dims(y_data, 1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

    return x_train, y_train, x_test, y_test


class HyperDataset(Dataset):
    def __init__(self, x_data, y_data, device='cpu', y_to_long=False):
        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data)
        self.y_data = self.y_data.long() if y_to_long else self.y_data.float()
        self.len = len(self.x_data)  # Size of data
        self.device = device

    def __getitem__(self, index):
        return self.x_data[index].to(self.device), self.y_data[index].to(self.device)

    def __len__(self):
        return self.len


class Net(nn.Module):
    def __init__(self, channels_num: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 21 * 23, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_data_train, y_data_train, x_data_test, y_data_test = get_data()

    batch_size = 4

    trainloader = torch.utils.data.DataLoader(
        dataset=HyperDataset(x_data=x_data_train, y_data=y_data_train, device=device),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    testloader = torch.utils.data.DataLoader(
        dataset=HyperDataset(x_data=x_data_train, y_data=y_data_train, device=device),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    classes = ('health', 'phyto')

    net = Net(channels_num=3)
    net.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # print(f"\noutputs\n{outputs}\n")
            # print(f"labels\n{labels}\n")
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}')

    print('Finished Training')
