from torchvision import models
from torch import nn, optim
import torch

def choose_architecture(arch_name, hidden_units, learning_rate, gpu=False):
    model = None

    if arch_name == 'efficientnet':
        model = models.efficientnet_b2(pretrained=True, weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False


        model.classifier = nn.Sequential(
            nn.Linear(1408, hidden_units),  # Increase the size of the first fully connected layer, replace n_inputs with 1408
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 102)  # Adjust the output size to match the number of classes, replace len(classes) with 102
        )

    elif arch_name == 'resnet':
        model = models.resnet50(pretrained=True, weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Linear(2048, hidden_units),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 102),
        )
        
    elif arch_name == 'vgg':
        model = models.vgg16(pretrained=True, weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 102),
        )
    else:
        print("Model Unknown.")
        exit()
    

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.96)

    if gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    return model, criterion, optimizer, step_scheduler