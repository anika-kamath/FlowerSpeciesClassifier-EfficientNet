from data_transforms import transform_data
from model_setup import choose_architecture

import torch
import numpy as np
from tqdm import tqdm
import argparse
import time, copy

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(data_dir, save_dir, arch, hidden_units, learning_rate, num_epochs, gpu=False):
    print("Architecture:", arch)
    print("Number of hidden units:", hidden_units)
    print("Learning rate:", learning_rate)
    print("Number of epochs:", num_epochs)
    if gpu:
        if torch.cuda.is_available():
            print("Using GPU.")
        else:
            print("GPU unavailable.")
            gpu = False
    else:
        if torch.cuda.is_available():
            print("To use GPU, set the --gpu flag, using CPU")

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

    train_loader, valid_loader, test_loader, train_data, valid_data = transform_data(train_dir, valid_dir, test_dir)

    model, criterion, optimizer, scheduler = choose_architecture(arch, hidden_units, learning_rate, gpu)

    training_history = {'accuracy':[],'loss':[]}
    validation_history = {'accuracy':[],'loss':[]}

    checkpoint_dir = 'checkpoints'
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
                inputs = inputs.to(device)
                labels = labels.to(device)

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

    # On test

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    test_loader_tqdm = tqdm(test_loader, total=len(test_loader), desc="Testing")

    with torch.no_grad():
        for inputs, labels in test_loader_tqdm:
            if gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

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

    checkpoint = {
        'architecture': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier': model.fc if arch == 'resnet' else model.classifier,
        'class_to_idx': model.class_to_idx,
        'epochs': num_epochs,
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(checkpoint, f'{save_dir}/checkpoint-{arch}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a new network on a dataset.')
    parser.add_argument('data_dir', type=str, help='Path to the directory.')
    parser.add_argument('--save_dir', type=str, default='checkpoints_folder', help='Path to the checkpoint directory.')
    parser.add_argument('--arch', type=str, default='efficientnet', help='Model architecture.')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')

    args = parser.parse_args()

    train_model(args.data_dir, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)