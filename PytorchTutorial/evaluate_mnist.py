import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model.tutorial_net import MnistTutorialNet
from data_manager import DataManager


if __name__ == "__main__":
    """
    This script load trained model and run on training set again
    ...
    just make sure the model learn something. we should not evaluate model on training set
    -- Meow --
    """
    model_path = "/data/cv_data/minist/mnistasjpg/saved_model8/MnistTutorialNetV5_80__0.19937348179519176.model"
    # at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define net
    net = MnistTutorialNet()
    # load model
    net.load_state_dict(torch.load(model_path))

    net = net.to(device) # cuda or cpu

    # load dataset
    imagenet_data = torchvision.datasets.ImageFolder(root='/data/cv_data/minist/mnistasjpg/trainingSet/',
                                                     transform=torchvision.transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=3)

    correct = 0
    total = 0
    with torch.no_grad():
        for datapoint in data_loader:
            images, labels = datapoint
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the  %d/%d images: %d %%' % (correct, total,
            100 * correct / total))