import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model.tutorial_net import MnistTutorialNet
from data_manager import DataManager

import torchvision
import torch

if __name__ == "__main__":

    # define net
    net = MnistTutorialNet()

    #
    imagenet_data = torchvision.datasets.ImageFolder(root='/data/cv_data/minis/mnistasjpg/trainingSet/',
                                                     transform=torchvision.transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=3)


    # define a loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
