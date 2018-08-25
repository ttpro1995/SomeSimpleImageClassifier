The code, ```main_mnist.py``` follow tutorial here (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

However, it use Mnist dataset as jpg (https://www.kaggle.com/scolianni/mnistasjpg).

In order to load image (not preprocessed file), use the following 

```python
import torchvision
import torch

imagenet_data = torchvision.datasets.ImageFolder(root='/data/cv_data/minist/mnistasjpg/trainingSet/',
                                                     transform=torchvision.transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=3)
```

The code, ```evaluate_mnist.py``` shows how to load trained model and call predict. 
