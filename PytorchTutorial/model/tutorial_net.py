import torch.nn as nn
import torch.nn.functional as F


class TutorialNet(nn.Module):
    def __init__(self):
        super(TutorialNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # cifar 4 3 32 32
        # mnist 4 3 28 28
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MnistTutorialNet(nn.Module):
    def __init__(self):
        super(MnistTutorialNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # cifar 4 3 32 32
        # mnist 4 3 28 28
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MnistTutorialNetV2(nn.Module):

    def __init__(self):
        super(MnistTutorialNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_mid = nn.Linear(84,84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout()


    def forward(self, x):
        # cifar 4 3 32 32
        # mnist 4 3 28 28
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.dropout(self.fc_mid(x)))
        x = F.log_softmax((self.fc3(x)))
        return x


class MnistTutorialNetV3(nn.Module):
    def __init__(self):
        super(MnistTutorialNetV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.fc1 = nn.Linear(16 * 3 * 3, 16 * 3 * 3)
        self.fc2 = nn.Linear(16 * 3 * 3, 120)
        self.fc_mid = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x):
        # cifar 4 3 32 32
        # mnist 4 3 28 28
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.dropout(self.fc_mid(x)))
        x = F.log_softmax((self.fc3(self.dropout(x))))
        return x


class MnistTutorialNetV4(nn.Module):
    def __init__(self):
        super(MnistTutorialNetV4, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 4 * 4, 16 * 3 * 3)
        self.fc2 = nn.Linear(16 * 3 * 3, 120)
        self.fc_mid_1 = nn.Linear(120, 100)
        self.fc_mid_2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x):
        # cifar 4 3 32 32
        # mnist 4 3 28 28
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc_mid_1(self.dropout(x)))
        x = F.relu(self.fc_mid_2(self.dropout(x)))
        x = F.log_softmax((self.fc3(self.dropout(x))))
        return x


class MnistTutorialNetV5(nn.Module):
    def __init__(self):
        super(MnistTutorialNetV5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 4 * 4, 16 * 3 * 3)
        self.fc2 = nn.Linear(16 * 3 * 3, 400)
        self.fc_mid_1 = nn.Linear(400, 320)
        self.fc_mid_2 = nn.Linear(320, 200)
        self.fc_mid_3 = nn.Linear(200, 184)
        self.fc3 = nn.Linear(184, 10)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        # cifar 4 3 32 32
        # mnist 4 3 28 28
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc_mid_1(self.dropout(x)))
        x = F.relu(self.fc_mid_2(self.dropout(x)))
        x = F.relu(self.fc_mid_3(self.dropout(x)))
        x = F.log_softmax((self.fc3(self.dropout(x))))
        return x


class MnistTutorialNetV6(nn.Module):
    def __init__(self):
        super(MnistTutorialNetV6, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 4 * 4, 16 * 3 * 3)
        self.fc2 = nn.Linear(16 * 3 * 3, 1000)
        self.fc_mid_1 = nn.Linear(1000, 700)
        self.fc_mid_2 = nn.Linear(700, 500)
        self.fc_mid_3 = nn.Linear(500, 350)
        self.fc_mid_4 = nn.Linear(350, 200)
        self.fc_mid_5 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        # cifar 4 3 32 32
        # mnist 4 3 28 28
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc_mid_1(self.dropout(x)))
        x = F.relu(self.fc_mid_2(self.dropout(x)))
        x = F.relu(self.fc_mid_3(self.dropout(x)))
        x = F.relu(self.fc_mid_4(self.dropout(x)))
        x = F.relu(self.fc_mid_5(self.dropout(x)))
        x = F.log_softmax((self.fc3(self.dropout(x))))
        return x