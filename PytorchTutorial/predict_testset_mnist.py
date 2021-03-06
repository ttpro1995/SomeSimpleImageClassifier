from image_folder_with_path import ImageFolderWithPaths
import torch
import torchvision
from model.tutorial_net import MnistTutorialNetV5
from util import get_img_id



if __name__ == "__main__":
    """
    this script predict testset for submit
    """
    # model_path = "/data/cv_data/minist/mnistasjpg/saved_model3/MnistTutorialNet_19__4.900279708264861.model"

    model_path = "/data/cv_data/minist/mnistasjpg/saved_model8/MnistTutorialNetV5_80__0.19937348179519176.model"

    # at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define net
    net = MnistTutorialNetV5()
    # load model
    net.load_state_dict(torch.load(model_path))

    net = net.to(device) # cuda or cpu

    # load dataset
    # data_dir = "/data/cv_data/minist/mnistasjpg/testSet"
    # imagenet_data = ImageFolderWithPaths(root=data_dir,
    #                                                  transform=torchvision.transforms.ToTensor())
    #
    # data_loader = torch.utils.DataLoader(imagenet_data,
    #                                           batch_size=4,
    #                                           num_workers=3)

    data_dir = "/data/cv_data/minist/mnistasjpg/wraptest/"
    train_dir = "/data/cv_data/minist/mnistasjpg/trainingSet/"
    dataset = ImageFolderWithPaths(data_dir, transform=torchvision.transforms.ToTensor())  # our custom dataset
    train_dir_for_class = ImageFolderWithPaths(train_dir, transform=torchvision.transforms.ToTensor())  # our custom dataset
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=40,
                                              num_workers=3)

    resultfile = open("result.csv", "w")
    resultfile.write("ImageId,Label\n")
    correct = 0
    total = 0
    with torch.no_grad():
        for datapoint in data_loader:
            images, _, path = datapoint
            images = images.to(device)
            # labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            for f, pred in zip(path, predicted):
                id = get_img_id(f)
                resultfile.write(str(id)+","+str(train_dir_for_class.classes[pred.item()]) + "\n")

    resultfile.close()
    print("done")