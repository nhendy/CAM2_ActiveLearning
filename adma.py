import os
import numpy as np
import numpy.linalg as LA
#§import cv2
from PIL import Image
import os
from collections import defaultdict
import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mean():
    classes = os.listdir('./train')
    center_images = defaultdict(str)


    for one_class in classes:
        min_error = None
        running_mean = 0
        center_image = ''

        images = os.listdir(os.path.join('train', one_class))
        for n, image in enumerate(images):
            img = Image.open(os.path.join('train', one_class, image)).convert('L')
            img2 = img.resize((224, 224))
            img_as_array = np.array(img2)

            running_mean = (1/(n + 1) ) * (img_as_array + n * running_mean)


        for image in images:
            img = Image.open(os.path.join('train', one_class, image)).convert('L')
            img2 = img.resize((224, 224))
            img_as_array = np.array(img2)

            current_error = LA.norm(img_as_array - running_mean)
            print(current_error)
            print(center_image)
            if(min_error is None or current_error < min_error):
                center_image = image
                min_error = current_error



        center_images[one_class] = center_image

        if not os.path.exists(os.path.join('centers', one_class)):
            os.makedirs(os.path.join('centers', one_class))

        try:
            Image.open(os.path.join('train', one_class, center_image)).save(os.path.join('centers', one_class, center_image))
        except:
            print('Error with class {} center image'.format(one_class))
    return (center_images)



print(get_mean())











