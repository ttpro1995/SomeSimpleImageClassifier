import torchvision
import torch
# https://stackoverflow.com/questions/50052295/how-do-you-load-images-into-pytorch-dataloader
imagenet_data = torchvision.datasets.ImageFolder(root='/data/cv_data/minis/mnistasjpg/trainingSet/',
                                                 transform=torchvision.transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=3)
for i, data in enumerate(data_loader, 0):
    inputs, labels = data
    print(inputs)
    print(labels)