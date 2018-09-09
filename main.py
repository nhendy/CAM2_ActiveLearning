from organize_data import rearrange_data
from transfer import train_model
from torchvision import datasets, models, transforms
import numpy as np
from torch.utils.data import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if  __name__ == '__main__':
    num_classes = rearrange_data()

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    medical_dataset = datasets.ImageFolder(root='./train',
                                           transform=data_transform)

    #  https://stackoverflow.com/questions/50544730/split-dataset-train-and-test-in-pytorch-using-custom-dataset
    #

    validation_split = 0.2
    shuffle_dataset = True
    batch_size = 16
    dataset_size = len(medical_dataset)
    random_seed = 42

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(medical_dataset, batch_size=batch_size,
                                               sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(medical_dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    dataloaders = {'train': train_loader, 'val': validation_loader}

    model_ft = models.resnet101(pretrained=True)
    freeze_layers = True


    if freeze_layers:
        for i, param in model_ft.named_parameters():
            param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    #
    #
    #
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)







