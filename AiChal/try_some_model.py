from torchvision.models.vgg import vgg19, vgg11
import torch
from util import get_train_valid_loader

# at beginning of the script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_conv = vgg11(pretrained=True)
model_conv = model_conv.to(device)
num_ftrs = model_conv.classifier[6].in_features

print(num_ftrs)
counter = 0
for name, child in model_conv.named_children():
    for name2, params in child.named_parameters():
        counter+=1
        if (counter < 11):
            params.requires_grad = False
            print(name, name2, "freeze")
        else:
            print(name, name2)

print(counter)