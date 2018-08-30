import torch.optim as optim
import torch.nn as nn
from model.tutorial_net import MnistTutorialNetV2
import os
from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor
from model.image_folder_with_path import ImageFolderWithPaths, remove_none_collate
import torchvision
import torch
from util import get_train_valid_loader

from torch.utils.data.dataloader import DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/5
if __name__ == "__main__":

    # at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define net
    net = MnistTutorialNetV2()
    net = net.to(device)  # cuda or cpu

    model_conv = torchvision.models.vgg11(pretrained=True)

    # Number of filters in the bottleneck layer
    num_ftrs = model_conv.classifier[6].in_features

    # convert all the layers to list and remove the last one
    features = list(model_conv.classifier.children())[:-1]

    ## Add the last layer based on the num of classes in our dataset
    features.extend([nn.Linear(num_ftrs, 103)])

    ## convert it into container and add it to our model class.
    model_conv.classifier = nn.Sequential(*features)


    landmark_input = "/data/cv_data/ai/again/TrainVal/"
    mnist_input = "/data/cv_data/minist/mnistasjpg/trainingSet/"

    #
    # imagenet_data = ImageFolderWithPaths(root=landmark_input,
    #                                      # transform=torchvision.transforms.ToTensor()
    #                                      transform=transforms.Compose([
    #                                          Resize((480, 480)),
    #                                          ToTensor()
    #                                      ])
    #                                      )

    train_loader, val_loader = get_train_valid_loader(landmark_input, 10, 113, 0.1, True, 1, True)



    # define a loss function
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-6)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            try:
                # get the inputs
                # mnist 100 3 28 28
                # landmark 100, 3, 480, 480
                if (data is None):
                    print("discard batch")
                    continue
                inputs, labels, paths = data
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
                if i % 100 == 99:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            except:
                print("discard batch")

        # save model # https://pytorch.org/docs/stable/notes/serialization.html
        save_path = os.path.join("/data/cv_data/ai/saved/landmark3",
                                 "vgg11" + str(epoch) + "__" + str(running_loss) + ".model")
        torch.save(net.state_dict(), save_path)
        print("saved " + save_path)

    print('Finished Training')
