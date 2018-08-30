import torch
import torchvision
from model.tutorial_net import MnistTutorialNetV2
from util import get_img_id
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from model.image_folder_with_path import ImageFolderWithPaths, remove_none_collate

if __name__ == "__main__":
    """
    This script load trained model and run on training set again
    ...
    just make sure the model learn something. we should not evaluate model on training set
    -- Meow --
    """
    model_path = "/data/cv_data/ai/saved/landmark1/MnistTutorialNetV2_10__925.3475708961487.model"
    # at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define net
    net = MnistTutorialNetV2()
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

    data_dir = "/data/cv_data/ai/testset"
    dataset = ImageFolderWithPaths(data_dir, transform=transforms.Compose([
                                             Resize((480, 480)),
                                             ToTensor()
                                         ]))  # our custom dataset
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=40,
                                              num_workers=4,
                                              collate_fn=remove_none_collate)

    resultfile = open("result.csv", "w")
    resultfile.write("id,predicted\n")
    correct = 0
    total = 0
    with torch.no_grad():
        for datapoint in data_loader:
            images, _, path = datapoint
            images = images.to(device)
            # labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            _, multi_preds = outputs.data.topk(3)
            for f, m_pred in zip(path, multi_preds):
                id = get_img_id(f)
                pending_str = str(id) + ","
                for pred in m_pred:
                    pending_str += str(pred.item()) + " "
                resultfile.write(pending_str + "\n")

    resultfile.close()
    print("done")