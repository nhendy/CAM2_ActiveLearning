# Active Learning

## Requirements
To install requirements run:

```
pip3 install -r requirements
```
## Dataset
This porject uses this [dataset](https://www.kaggle.com/nih-chest-xrays/data/home) from kaggle. For using the sample dataset which is 2GB run: 
```
kaggle datasets download -d nih-chest-xrays/sample
```
To download the full dataset which is 42GB run:
```
kaggle datasets download -d nih-chest-xrays/data
```

The full dataset has 112,120 X-ray images with labels from 30,805 unique patients.

## Dataset Limitations
* The image labels are NLP extracted so there could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%.

## Using Kaggle
To use kaggle API follow [this guide](https://github.com/Kaggle/kaggle-api).

## To organize dataset

Run

```python
python3 organize_data.py
```

A train folder will be created with this tree structure
```
.
├── Atelectasis
├── Cardiomegaly
├── Consolidation
├── Edema
├── Effusion
├── Emphysema
├── Fibrosis
├── Hernia
├── Infiltration
├── Mass
├── No\ Finding
├── Nodule
├── Pleural_Thickening
├── Pneumonia
└── Pneumothorax
```
Each directory will have images of corresponding class name.
