# Benchmarking Image Feature Extraction on Lung Cancer Classification

## Introduction

---

The purpose of this project was to evaluate the effectiveness of scikit-image feature extraction for image classification tasks. 

A popular example of applied image classification is in the field of Cancer Histopathology. This field makes heavy use of Convolutional Neural Networks in order to train a model that can discriminate between different types of cancer and benign tissue. 

The scikit-image package is a powerful library in Python that provides a wide range of tools for image processing and feature extraction. It is built on top of other scientific libraries like NumPy and SciPy, making it highly efficient for image manipulation and analysis. Using this package, one can convert an images into tabular data.

Using the inception v3 model as a benchmark, this project aims to evaluate the effectiveness of the extracted features as a possible way to reduce the dimensionality of image classification tasks while maintaining adequate performance.

## Datasets

### Original Dataset

The original dataset can be found on kaggle [here.](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data) It includes 25000 images split between the following classes:
- Lung benign tissue
- Lung adenocarcinoma
- Lung squamous cell carcinoma
- Colon adenocarcinoma
- Colon benign tissue

Citation:
Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019

### Subset

For this project, only the lung cancer classes were used. A subset of the data can be found on dropbox at this [link.](https://www.dropbox.com/scl/fi/arkoncxpl5y1dzmviivi2/lung_image_set.tar.gz?rlkey=716vzb9dyssledyxojmxy0kr0&st=qw4unm62&dl=1)

To use this dataset you can either upload it to the project directory or run the following code:
```python
import os
import requests
import tarfile

# URL to download your folder
url = "https://www.dropbox.com/scl/fi/arkoncxpl5y1dzmviivi2/lung_image_set.tar.gz?rlkey=716vzb9dyssledyxojmxy0kr0&st=qw4unm62&dl=1"

output_path = "/content/lung_image_set.tar.gz"

# Download the GZip folder
response = requests.get(url)
print(f"HTTP Response Code: {response.status_code}")

if response.status_code == 200:
    print("Download successful")
else:
    print(f"Failed to download file. HTTP status code: {response.status_code}")

with open(output_path, "wb") as f:
    f.write(response.content)
```

Then extract with:

```python
import tarfile

output_path = "/content/lung_image_set.tar.gz"
extract_path = "/content/"

# Extract the .tar.gz archive
if tarfile.is_tarfile(output_path):
    with tarfile.open(output_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Extraction complete")
else:
    print(f"{output_path} is not a valid .tar.gz archive")
```

This should leave you with a lung_image_sets folder with 3 subdirectories.

### Tabular Extracted Features

This dataset is in the repo named image_features.csv. It is the result of processing all 15000 images using the `scikit-image` package.

## Notebooks

### Preprocessing_and_Benchmarking

This notebook walks through the creation of the tabular dataset and the selected features to extract from each image. It also conducts the inception v3 transfer learning benchmark.

### Classification_on_Tabular_Data

This notebook conducts the preprocessing, training, and evaluation of the tabular dataset created in Preprocessing_and_Benchmarking.

## Requirements

---

## Dependencies

### Environment
* Python 3.10.12: Target version for compatability and performance
  pandas: A powerful data manipulation and analysis library, providing data structures like DataFrames for efficient handling of large datasets.
* Google CoLab: A cloud-based Jupyter notebook environment with GPUs for efficient model training.

### Important Packages

* scikit-image: A collection of image processing algorithms built on top of SciPy, used to extract features from images and create a tabular dataset.

* scikit-learn: A machine learning library primarily used for data processing for this project

* torch: The core PyTorch library, providing support for deep learning and tensor computations, including automatic differentiation and GPU acceleration.

* torchvision: A package that provides datasets, model architectures, and image transformations for deep learning in computer vision, built on top of PyTorch.

* pytorch-tabnet: A deep learning library built on PyTorch for tabular data, utilizing a novel attention-based architecture for efficient and interpretable predictions.

* seaborn: A Python visualization library based on Matplotlib, which was used to visualize the mode evaluations.

Add dependencies using `!pip install -r requirements.txt` in the notebook.

