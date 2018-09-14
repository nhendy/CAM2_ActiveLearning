# -*- coding: utf-8 -*-
"""
Transfer Learning tutorial
==========================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

In this tutorial, you will learn how to train your network using
transfer learning. You can read more about the transfer learning at `cs231n
notes <http://cs231n.github.io/transfer-learning/>`__

Quoting these notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios look as follows:

-  **Finetuning the convnet**: Instead of random initializaion, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

"""
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import SubsetRandomSampler
from organize_data import rearrange_data

import matplotlib.pyplot as plt
import time
import os
import copy

#plt.ion()   # interactive mode



######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated





######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0



    #TODO
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    training_accuracy_list = []
    val_accuracy_list = []


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == "__main__":

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    # The problem we're going to solve today is to train a model to classify
    # **ants** and **bees**. We have about 120 training images each for ants and bees.
    # There are 75 validation images for each class. Usually, this is a very
    # small dataset to generalize upon, if trained from scratch. Since we
    # are using transfer learning, we should be able to generalize reasonably
    # well.
    #
    # This dataset is a very small subset of imagenet.
    #
    # .. Note ::
    #    Download the data from
    #    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
    #    and extract it to the current directory.

    # Data augmentation and normalization for training
    # Just normalization for validation

    #For train folder with eval folder
    '''
    with_eval_folder = 1
    data_dir = 'hymenoptera_data'#'Medical_data'##

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if with_eval_folder:


        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        print(type(image_datasets['train']))
        class_names = image_datasets['train'].classes
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        print('with eval folder')

    #For train folder without eval folder
    else:
        full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        class_names = full_dataset.classes 
        print(class_names)  
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        dataset_sizes = {x: train_size if x=='train' else test_size for x in ['train', 'val'] }
        image_datasets = {x: train_dataset if x=='train' else test_dataset for x in ['train', 'val'] }
        print(image_datasets)
        print('without eval folder')


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Get a batch of training data
    #inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    #out = torchvision.utils.make_grid(inputs)

    #imshow(out, title=[class_names[x] for x in classes])
'''

#Credit to nhendy


    if not (os.path.exists('./Medical_data/train')):
        rearrange_data()


    num_classes = len(list(os.walk('./Medical_data/train'))) - 1

    print("number of classes is : {}".format(num_classes))

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    medical_dataset = datasets.ImageFolder(root='./Medical_data/train',
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model_ft = models.resnet101(pretrained=True)
    freeze_layers = True


    if freeze_layers:
        for i, param in model_ft.named_parameters():
            param.requires_grad = False
    #
    num_ftrs = model_ft.fc.in_features
    print(num_ftrs)
    # print(type(num_ftrs))
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    # #
    # #
    # #
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device,
                           num_epochs=25)

    torch.save(model_ft.state_dict(), 'resnet101_5K_medical.pt')
    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrained model and reset final fully connected layer.
    #
'''
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 15-25 min on CPU. On GPU though, it takes less than a
    # minute.
    #

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)

    torch.save(model_ft.state_dict(), "new_trained.pt")
    #with open('new_trained_weight.txt', 'w') as file:
    #    file.write(model_ft.fc2.weight.data)

    ######################################################################
    #

    #visualize_model(model_ft)


    ######################################################################
    # ConvNet as fixed feature extractor
    # ----------------------------------
    #
    # Here, we need to freeze all the network except the final layer. We need
    # to set ``requires_grad == False`` to freeze the parameters so that the
    # gradients are not computed in ``backward()``.
    #
    # You can read more about this in the documentation
    # `here <http://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
    #

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # On CPU this will take about half the time compared to previous scenario.
    # This is expected as gradients don't need to be computed for most of the
    # network. However, forward does need to be computed.
    #

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25)
    torch.save(model_conv.state_dict(), "final_trained.pt")
    #with open('final_trained_weight.txt', 'w') as file:
    #    file.write(model_conv.fc2.weight.data)
    ######################################################################
    #

    #visualize_model(model_conv)

    #plt.ioff()
    #plt.show()
'''

