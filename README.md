
# Capstone

# Classifying Marine Vessels using Convolutional Neural Networks
**Author: Lera Tsayukova**

![example](

## Overview
The Problem: 
Among growing tension worldwide, there are a range of problems surfacing off the shores of virtually every ocean-adjacent country. 
These problems include overfishing, piracy, military encroachment, and the violation of exclusive economic zones. In other words issues that span across socio-economic and military sectors.
Naturally, the ability to recognize and distinguish different kinds of ships can go a long way in preventing or at least more efficently dealing with these types of problems. 
The goal of this project is to be able to correctly identify the type of marine vessel based on the image.  
Two types of models will be created including 
Multi Layer Perceptron (MLP) models as well as Convolutional Neural Network (CNN) models. These models will be evalued using accuracy scores to stress the 
importance of minimizing false positives.

## Methods
The methodology for this project includes data cleaning, exploratory data analysis, feature engineering, and running various models to determine the model 
with the highest accuracy score.

## Data
The data consisted of a total of 5,252 images from Kaggle. The target variable is broken down into 5 categories.
**List dictionary here**
The images were processed using Keras' image processing tools, including Keras Image Data Generator.

## Exploratory Data Analysis
To become familiar with the data, images from each category in the training set are visualized to see what they look like respectively. some images in the normal train set as well as the pneumonia train set are visualized to see what the images look like. 
Furthermore, a graph is generateed to explore class imbalance within the data.

### Visualization
The images are visualized to compare the different types of classes included in the dataset. The difference are fairly distinct among classes, 
except cargo ships and carrier ships which may be more challenging for the models to distinguish.


**Class 1:Cargo**

![example](images/normal.png)

**Class 2: Military**

![example](images/pneumonia.png)

**Class 3: Carrier**

![example](images/pneumonia.png)

**Class 4: Cruise**

![example](images/pneumonia.png)

**Class 5: Tanker**

![example](images/pneumonia.png)



### Class Imbalance
In plotting the counts for each of the five classes of ships, we can see a slight class imbalance between the five categories with more of class 1: Cargo, then the rest. To rectify this, 
a class weight is created to use when modeling.

![class_imbalance](https://user-images.githubusercontent.com/75099138/127192651-9c8b5770-c0e0-4fc7-a31d-4c630a7e9960.png)

## Models
Various models are generated in order to find a model with the best accuracy score. The MLP (MultiLayerPerceptron) models are created first, followed by the CNN (Convultion Neural Net) models. There was a high amount of
overfitting in most models so regularization techniques including l1 regularization, early stopping, and dropout (l2) are implemented in order to minimize overfitting. 
The best model was the third CNN model which included all three regularization techniques.

## Final Model
The final model included an l1 of 0.0005, activation of 'relu' and 'sigmoid', dropout of 0.1, and early stopping. With this model the recall on the train set 
is 0.9918, recall on the validation set is 0.8034, and recall on the test set is 1. 

![example](images/final_model_graph.png)

![example](images/validation_cm.png)

![example](images/test_cm.png)

The confusion matrix shows that the model does have more false positives. However, because the focus was to create a model which minimizes false negatives, this model is still deemed the best model for determining if a 
the class of ship.

## Next Steps: 
Some future steps to improve the model include:
  - Running more models to try and improve Accuracy score
  - Trying other techniques to minimize overfitting, such as batch normalization
  - Implementing transfer learning models

## Repository Structure
```
├── best_models
├── data
├── images
├── notebooks
├── Final_Notebook.ipynb
├── Presentation.pdf
└── README.md
