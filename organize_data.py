import pandas
import os
import shutil
import torch
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import SubsetRandomSampler

src_path = './images/'
new_path = './train/'


def rearrange_data():
    df = pandas.read_csv('sample_labels.csv', index_col='Image Index')
    labels = df['Finding Labels']

    classes = set()

    for image, label in labels.items():
        actualabels = label.split('|')
        for actualabel in actualabels:
            target_path = new_path + actualabel.replace(" ", "")
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            shutil.copyfile(src_path + image, target_path + '/' + image)
            classes.add(actualabel)

    return classes


if __name__ == '__main__':
    rearrange_data()
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    medical_dataset = datasets.ImageFolder(root='./train',
                                           transform=data_transform)




    #  https://stackoverflow.com/questions/50544730/split-dataset-train-and-test-in-pytorch-using-custom-dataset

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


