import torch
from torch.utils.data.dataloader import default_collate
from torchvision import datasets


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        try:
            # this is what ImageFolder normally returns
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path
        except Exception as e:
            print (e)


def remove_none_collate(batch):
    try:
        batch = list(filter(lambda x:x is not None, batch))
        ret = default_collate(batch)
        return ret
    except:
        print("error at remove none collate")
        ret = None
        return ret


def example():
    # instantiate the dataset and dataloader
    data_dir = "/data/cv_data/minist/mnistasjpg/testSet"
    dataset = ImageFolderWithPaths(data_dir)  # our custom dataset
    dataloader = torch.utils.DataLoader(dataset)

    # iterate over data
    for inputs, labels, paths in dataloader:
    # use the above variables freely
        print(inputs, labels, paths)