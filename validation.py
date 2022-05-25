import torch
from torchvision import transforms

from networks.resnet_big import SupConResNet
from custom_dataset import CustomDataset
from util import TwoCropTransform


def validate_model(path: str):
    checkpoint = torch.load(path)
    opt = checkpoint["opt"]

    model = SupConResNet(name=opt.model)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if opt.dataset == 'path':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    image_folder = opt.data_folder+"images/test"
    texts_data = opt.data_folder+"texts/test_titles.csv"
    test_dataset = CustomDataset(image_folder, texts_data, transform=TwoCropTransform(train_transform))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

    for idx, (images, descriptions, labels) in enumerate(test_loader):
        images = torch.cat([images[0], images[1]], dim=0)
        bsz = labels.shape[0]

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)


        # compute loss
        features, text_features = model(images, descriptions)