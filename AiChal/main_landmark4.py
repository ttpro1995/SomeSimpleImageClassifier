import torch.optim as optim
import torch.nn as nn
from model.tutorial_net import MnistTutorialNetV2
import os
from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor
from model.image_folder_with_path import ImageFolderWithPaths, remove_none_collate
import torchvision
import torch
from util import get_train_valid_loader_v2, vgg_transformation

from torch.utils.data.dataloader import DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/5
if __name__ == "__main__":

    # at beginning of the script
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # filepath
    landmark_input = "/data/cv_data/TrainVal"
    model_base_path = "/home/zdeploy/thient/model/landmark/landmark3"

    # landmark_input = "/data/voice_zaloai/recognition/train/"
    # model_base_path = "/home/zdeploy/thient/model/landmark/landmark3"
    print("path----------")
    print(landmark_input)
    print(model_base_path)
    print("===============")


    # define net
    # net = MnistTutorialNetV2()

    model_conv = torchvision.models.vgg16(pretrained=True)

    # freeze some layer
    counter = 0
    for name, child in model_conv.named_children():
        for name2, params in child.named_parameters():
            counter += 1
            if (counter < 27):
                params.requires_grad = False
                print(name, name2, "freeze")
            else:
                print(name, name2)

    # Number of filters in the bottleneck layer
    num_ftrs = model_conv.classifier[6].in_features

    # convert all the layers to list and remove the last one
    features = list(model_conv.classifier.children())[:-1]

    ## Add the last layer based on the num of classes in our dataset
    features.extend([nn.Linear(num_ftrs, 103)])

    ## convert it into container and add it to our model class.
    model_conv.classifier = nn.Sequential(*features)

    net = model_conv.to(device)  # cuda or cpu

    # for vgg model
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform, val_transform = vgg_transformation(normalize=normalize)

    train_loader, val_loader = get_train_valid_loader_v2(landmark_input, 2, 113,
                                                         train_transform, val_transform,
                                                         0.1, True, 1, True)



    # define a loss function
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-2)

    for epoch in range(2000):  # loop over the dataset multiple times
        net.train()
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

        val_correct = 0
        val_total = 0

        net.eval()
        for datapoint in val_loader:
            images, labels, path = datapoint
            images = images.to(device)
            # labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            corrects = torch.sum(predicted.to("cpu") == labels.data)
            total = len(labels.data)
            val_correct += corrects.item()
            val_total += total

        print(val_correct)
        print(val_total)
        print(1.0 * val_correct/ val_total)

        # save model # https://pytorch.org/docs/stable/notes/serialization.html
        save_path = os.path.join(model_base_path,
                                 "vgg11" + str(epoch) + "_" + str(running_loss) + ".model")
        torch.save(net.state_dict(), save_path)
        print("saved " + save_path)

    print('Finished Training')
