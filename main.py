from organize_data import rearrange_data
from transfer import train_model
from torchvision import datasets, models, transforms
import numpy as np
from torch.utils.data import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if  __name__ == '__main__':

    if not (os.path.exists('./train')):
        rearrange_data()


    num_classes = len(list(os.walk('./train'))) - 1

    print("number of classes is : {}".format(num_classes))

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

    # 20% 10% 20% 30%
    validation_split = 0.2
    first_split = 0.2
    second_split = 0.2
    third_split = 0.2
    # fourth_split = 0.2

    shuffle_dataset = True
    batch_size = 20
    dataset_size = len(medical_dataset)
    random_seed = 42

    indices = list(range(dataset_size))

    val_split = int(np.floor(validation_split * dataset_size))
    first_end = 2*val_split
    second_end = 3 * val_split
    third_end = 4 * val_split
    # fourth_split = int(np.floor(fourth_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    val_indices, first_indices, second_indices, third_indices, fourth_indices  = indices[:val_split], \
                                                                                 indices[val_split: first_end],\
                                                                                 indices[first_end: second_end],\
                                                                                 indices[second_end: third_end],\
                                                                                 indices[third_end: ]




    first_sampler = SubsetRandomSampler(first_indices)
    second_sampler = SubsetRandomSampler(second_indices)
    third_sampler = SubsetRandomSampler(third_indices)
    fourth_sampler = SubsetRandomSampler(fourth_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    first_loader = torch.utils.data.DataLoader(medical_dataset, batch_size=batch_size,
                                               sampler=first_sampler)
    second_loader = torch.utils.data.DataLoader(medical_dataset, batch_size=batch_size,
                                               sampler=second_sampler)
    third_loader = torch.utils.data.DataLoader(medical_dataset, batch_size=batch_size,
                                                sampler=third_sampler)
    fourth_loader = torch.utils.data.DataLoader(medical_dataset, batch_size=batch_size,
                                                sampler=fourth_sampler)

    validation_loader = torch.utils.data.DataLoader(medical_dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    dataloaders_first = {'train': first_loader, 'val': validation_loader}
    dataloaders_second = {'train': second_loader, 'val': validation_loader}
    dataloaders_third = {'train': third_loader, 'val': validation_loader}
    dataloaders_fourth = {'train': fourth_loader, 'val': validation_loader}


    all_dataloaders = [dataloaders_first, dataloaders_second, dataloaders_third, dataloaders_fourth]

    model_ft = models.resnet101(pretrained=True)
    freeze_layers = True


    if freeze_layers:
        for i, param in model_ft.named_parameters():
            param.requires_grad = False
    #
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #save initial weights
    torch.save(model_ft.state_dict(),'models/resnet_initial.pt')
    output_path = 'logs/log'

    for i, loader in enumerate(all_dataloaders):


        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, loader, device, output_path + i,
                           num_epochs=25)
        torch.save(model_ft.state_dict(), 'models/resnet101_5K_medical_loader_' + i + '.pt')

        model_ft.load_state_dict(torch.load('models/resnet_initial.pt'))









