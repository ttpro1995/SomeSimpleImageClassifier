import torch.optim as optim
import torch.nn as nn
from model.tutorial_net import MnistTutorialNet
import os

import torchvision
import torch

if __name__ == "__main__":

    # at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define net
    net = MnistTutorialNet()
    net = net.to(device) # cuda or cpu

    #
    imagenet_data = torchvision.datasets.ImageFolder(root='/data/cv_data/minist/mnistasjpg/trainingSet/',
                                                     transform=torchvision.transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=100,
                                              shuffle=True,
                                              num_workers=3)


    # define a loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

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


        # save model # https://pytorch.org/docs/stable/notes/serialization.html
        save_path = os.path.join("/data/cv_data/minist/mnistasjpg/saved_model2/",
                                 "MnistTutorialNet_" + str(epoch) + "__" + str(running_loss) + ".model")
        torch.save(net.state_dict(), save_path)
        print("saved " + save_path)

    print('Finished Training')
