class MnistDataLoader:
    def __init__(self):
        imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
        data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                  batch_size=4,
                                                  shuffle=True,
                                                  num_workers=args.nThreads)