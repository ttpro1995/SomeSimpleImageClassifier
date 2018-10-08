import torch.optim as optim
import torch.nn as nn
from model.tutorial_net import MnistTutorialNetV6
import os
from torch.utils.data import random_split
import torchvision
import torch
import configparser

if __name__ == "__main__":
    """
    with train and dev set
    """
    config = configparser.ConfigParser()
    config.read("config/mnistdesktop.ini")

    dataset_path = config["MNIST"]["dataset"]
    model_path = config["MNIST"]["modeldir"]

    # at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define net
    net = MnistTutorialNetV6()
    net = net.to(device)  # cuda or cpu

    #
    raw_dataset = torchvision.datasets.ImageFolder(root=dataset_path,
                                                     transform=torchvision.transforms.ToTensor())
    train_size = int(0.8 * len(raw_dataset))
    test_size = len(raw_dataset) - train_size
    print("Train size " + str(train_size))
    print("Dev size " + str(test_size))
    training_set, dev_set = random_split(raw_dataset, [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(training_set,
                                              batch_size=2000,
                                              shuffle=True,
                                              num_workers=4)

    dev_data_loader = torch.utils.data.DataLoader(dev_set,
                                                    batch_size=2000,
                                                    shuffle=True,
                                                    num_workers=4)


    # define a loss function
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-3, lr=1e-3)

    for epoch in range(100):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
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
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        #######################################################################

        with torch.no_grad():
            net.eval()
            correct = 0
            total = 0
            for datapoint in train_data_loader:
                images, labels = datapoint
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("epoch " + str(epoch))
            print('Accuracy of the network on the train set %d/%d images: %f %%' % (correct, total,
                                                                           100.0 * correct / total))
            correct = 0
            total = 0
            for datapoint in dev_data_loader:
                images, labels = datapoint
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("epoch " + str(epoch))
            print('Accuracy of the network on the dev set %d/%d images: %f %%' % (correct, total,
                                                                           100.0 * correct / total))

        #######################################################################
        # save model # https://pytorch.org/docs/stable/notes/serialization.html
        if epoch > 50 and epoch %2 == 0:
            save_path = os.path.join(model_path,
                                     "MnistTutorialNetV6_" + str(epoch) + "_" + str(100.0 * correct / total) + ".model")
            torch.save(net.state_dict(), save_path)
            print("saved " + save_path)

    print('Finished Training')
