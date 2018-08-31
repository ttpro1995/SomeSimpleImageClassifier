import torch.optim as optim
import torch.nn as nn
import sys
from model.tutorial_net import MnistTutorialNetV2
import os
from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor
from model.image_folder_with_path import ImageFolderWithPaths, remove_none_collate
import torchvision
import torch
from util import get_train_valid_loader
from util import get_img_id
from torch.utils.data.dataloader import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/5
if __name__ == "__main__":
    model_path = sys.argv[1]
    # at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # filepath
    landmark_input = "/data/cv_data/TrainVal"
    # model_base_path = "/home/zdeploy/thient/model/landmark/landmark1"

    # landmark_input = "/data/voice_zaloai/recognition/test"
    # model_base_path = "/home/zdeploy/thient/model/landmark/landmark1"
    print("path----------")
    print(landmark_input)
    print(model_path)
    print("===============")


    # define net
    # net = MnistTutorialNetV2()

    model_conv = torchvision.models.vgg11(pretrained=True)

    # Number of filters in the bottleneck layer
    num_ftrs = model_conv.classifier[6].in_features

    # convert all the layers to list and remove the last one
    features = list(model_conv.classifier.children())[:-1]

    ## Add the last layer based on the num of classes in our dataset
    features.extend([nn.Linear(num_ftrs, 103)])

    ## convert it into container and add it to our model class.
    model_conv.classifier = nn.Sequential(*features)

    net = model_conv.to(device)  # cuda or cpu
    net.load_state_dict(torch.load(model_path))

    # for vgg model
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if (normalize is not None):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        val_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    dataset = ImageFolderWithPaths(landmark_input, transform=val_transform)  # our custom dataset
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=40,
                                              num_workers=4,
                                              collate_fn=remove_none_collate)

    val_correct = 0
    val_total = 0
    net.eval()
    resultfile = open("resultlandmark2.csv", "w")
    resultfile.write("id,predicted\n")
    with torch.no_grad():
        net.eval()
        for datapoint in data_loader:
            images, _, path = datapoint
            images = images.to(device)
            # labels = labels.to(device)
            outputs = net(images)
            # _, predicted = torch.max(outputs.data, 1)
            _, multi_preds = outputs.data.topk(3)
            for f, m_pred in zip(path, multi_preds):
                id = get_img_id(f)
                pending_str = str(id) + ","
                for pred in m_pred:
                    pending_str += str(pred.item()) + " "
                resultfile.write(pending_str + "\n")


    print('Finished predict')
