import torch.optim as optim
import torch.nn as nn
from model.tutorial_net import MnistTutorialNetV2
import os
from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor
from model.image_folder_with_path import ImageFolderWithPaths, remove_none_collate
import torchvision
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/5
if __name__ == "__main__":

    # at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define net
    net = MnistTutorialNetV2()
    net = net.to(device) # cuda or cpu

    landmark_input = "/data/cv_data/ai/again/TrainVal/"
    mnist_input = "/data/cv_data/minist/mnistasjpg/trainingSet/"

    #
    imagenet_data = ImageFolderWithPaths(root=landmark_input,
                                         # transform=torchvision.transforms.ToTensor()
                                         transform=transforms.Compose([
                                             Resize((480, 480)),
                                             ToTensor()
                                         ])
                                         )

    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=40,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=remove_none_collate)

    
    # define a loss function
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-6)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            # mnist 100 3 28 28
            # landmark 100, 3, 480, 480
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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


        # save model # https://pytorch.org/docs/stable/notes/serialization.html
        save_path = os.path.join("/data/cv_data/ai/saved/landmark1",
                                 "MnistTutorialNetV2_" + str(epoch) + "__" + str(running_loss) + ".model")
        torch.save(net.state_dict(), save_path)
        print("saved " + save_path)

    print('Finished Training')
