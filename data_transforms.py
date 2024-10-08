import torch 
from torchvision import datasets, transforms
# TODO: Define your transforms for the training, validation, and testing sets
import torchvision.transforms as transforms

def transform_data(train_dir, valid_dir, test_dir):
    # common transformation steps
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define specific transformations for each set
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        data_transforms
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        data_transforms
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        data_transforms
    ])

    # TODO: Load the datasets with ImageFolder
    # all_data = datasets.ImageFolder(data_dir, transform = data_transforms)
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    image_datasets = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloader = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(valid_data, batch_size=64),
        'test': torch.utils.data.DataLoader(test_data, batch_size=64)
    }

    train_loader = dataloader['train']
    test_loader = dataloader['test']
    valid_loader = dataloader['valid']
    return train_loader, valid_loader, test_loader, train_data, valid_data
