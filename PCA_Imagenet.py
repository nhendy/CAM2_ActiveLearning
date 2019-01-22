import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np
from glob import glob
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

#Input the Data folder here
BASE_DATA_FOLDER = "../home/data/ilsvrc/ILSVRC"#"../cat-and-dog"#../hymenoptera_data"#"../Medical_data"
TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "ILSVRC2012_Classification")

#Plot
def visualize_scatter(data_2d, label_ids, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    
    for label_id in np.unique(label_ids):
        if (id_to_label_dict[label_id] == "highlight"):
            plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                        data_2d[np.where(label_ids == label_id), 1],
                        marker='o',
                        color= 'red',
                        linewidth='1',
                        alpha=0.5,
                        label=id_to_label_dict[label_id])
        else:
            plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                        data_2d[np.where(label_ids == label_id), 1],
                        marker='o',
                        color=  plt.cm.Set1(label_id / float(nb_classes)),
                        linewidth='1',
                        alpha=0.5,
                        label=id_to_label_dict[label_id])

    plt.legend(loc='best')
    plt.show() 

images = []
labels = []

#First resize the image to 200*200 with grey scale
for class_folder_name in os.listdir(TRAin_DATA_FOLDER):
    class_folder_path = os.path.join(TRAin_DATA_FOLDER, class_folder_name)
    for image_path in glob(os.path.join(class_folder_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (100, 100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.flatten()
        images.append(image)
        labels.append(class_folder_name)
        
images = np.array(images)
labels = np.array(labels)

#Make the order of the label names
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

#Normalize the data
label_ids = np.array([label_to_id_dict[x] for x in labels])
images_scaled = StandardScaler().fit_transform(images)

#print(images_scaled.shape)
#plt.imshow(np.reshape(images[35], (200,200)), cmap="gray")

#visualize_scatter(images_scaled , label_ids)


#How many features
pca = PCA(n_components=2)
pca_result = pca.fit_transform(images_scaled)
pca_result_scaled = StandardScaler().fit_transform(pca_result)
visualize_scatter(pca_result_scaled, label_ids)
#print(pca.explained_variance_ratio_)

#print(pca_result.shape)
#Based on https://distill.pub/2016/misread-tsne/, the perplexity value can significantly affect the output plot 

#tsne = TSNE(n_components=2, perplexity=40.0)
#tsne_result = tsne.fit_transform(pca_result)
#tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
#visualize_scatter(tsne_result_scaled, label_ids)

