import pandas
import os
import shutil

src_path='./images/'
new_path = './train/'

def main():
    df = pandas.read_csv('sample_labels.csv', index_col='Image Index')
    labels = df['Finding Labels']

    classes = set()

    for image, label in labels.items():
        actualabels = label.split('|')
        for actualabel in actualabels:
            target_path = new_path + actualabel
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            shutil.copyfile(src_path + image, target_path + '/' + image)
            classes.add(actualabel)

    return classes





if __name__ == '__main__':
    main()