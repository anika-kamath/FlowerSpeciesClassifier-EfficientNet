import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
import torchvision.transforms as models
import time
import copy

import argparse
import os


def choose_architecture(arch_name, hidden_units, learning_rate, num_epochs, gpu=False):
    model = None

    if arch_name == 'efficientnet':
        model = models.efficientnet_b2(weights="IMAGENET1K_V1", pretrained=True)

        #freeze the pretrained layers
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Linear(1408, hidden_units),  # Increase the size of the first fully connected layer
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 102)  # Adjust the output size to match the number of classes (102 == len(classes))
        )
    
    elif arch_name == 'resnet':
        model = models.resnet50(weights="IMAGENET1K_V1", pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Linear(2048, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 102),
        )

    elif arch_name == 'vgg':
        model = models.vgg16(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 102),
        )
    
    else:
        print(f'Unknown architecture: {arch_name}')
        exit()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.96)

    if gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    return model, criterion, optimizer, step_scheduler
        
    

def transform_data(train_dir, valid_dir, test_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
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

def train_model(data_dir, save_dir, arch, hidden_units, learning_rate, num_epochs, gpu=False):
    print("Using model architecture:", arch)
    print("Number of hidden units:", hidden_units)
    print("Learning rate:", learning_rate)
    print("Number of epochs:", num_epochs)


    if gpu:
        if torch.cuda.is_available():
            print("Using GPU for training.")
        else:
            print("GPU is unavailable, using CPU.")
            gpu = False
    else:
        if torch.cuda.is_available():
            print("GPU is available but not selected for training. To use GPU, set the --gpu flag.")
        print("Using CPU for training.")

    train_dir = f'{data_dir}/train'
    valid_dir = f'{data_dir}/valid'
    test_dir = f'{data_dir}/test'

    dataloaders = {
        "train": train_loader,
        "val": valid_loader
    }

    dataset_sizes = {
        "train": len(train_data),
        "val": len(valid_data)
    }

    training_history = {'accuracy':[],'loss':[]}
    validation_history = {'accuracy':[],'loss':[]}

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)


    train_loader, valid_loader, test_loader, train_data, valid_data = transform_data(train_dir, valid_dir, test_dir)

    model, criterion, optimizer, scheduler = choose_architecture(arch, hidden_units, learning_rate, gpu)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                training_history['accuracy'].append(torch.tensor(epoch_acc).cpu())  # Convert to tensor and move to CPU
                training_history['loss'].append(epoch_loss)
            elif phase == 'val':
                validation_history['accuracy'].append(torch.tensor(epoch_acc).cpu())  # Convert to tensor and move to CPU
                validation_history['loss'].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'training_history': training_history,
            'validation_history': validation_history,
            
        }, checkpoint_path)

        print("Checkpoint saved:", checkpoint_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)


    # TODO: Do validation on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    test_loader_tqdm = tqdm(test_loader, total=len(test_loader), desc="Testing the network")

    with torch.no_grad():
        for inputs, labels in test_loader_tqdm:
            # inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loader_tqdm.set_postfix(loss=test_loss / len(test_loader), accuracy=100 * correct / total)

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    model.class_to_idx = train_data.class_to_idx


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_dir', type=str, help='Path to the data directory.')
    parser.add_argument('--save_dir', type=str, default='checkpoints_folder', help='Path to the checkpoint directory.')
    parser.add_argument('--arch', type=str, default='efficientnet', help='Model architecture.')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')

    args = parser.parse_args()

    train_model(args.data_dir, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)
