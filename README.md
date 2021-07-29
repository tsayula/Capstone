
## Capstone Project

# Classifying Marine Vessels using Convolutional Neural Networks
**Author: Lera Tsayukova**

![599881232](https://user-images.githubusercontent.com/75099138/127193063-4bdc8067-78b2-4a0a-b489-1412f57caafc.jpeg)


## Overview
**Business Problem:**
Among growing tension worldwide, there are a range of problems surfacing off the shores of virtually every ocean-adjacent country. 
This include overfishing, piracy, military encroachment, and the violation of exclusive economic zones. Billions of dollars are lost every year in an effort to combat these issues.  
**Use Case:**
Naturally, the ability to recognize and distinguish different kinds of ships can go a long way in preventing or at least more efficently dealing with these types of problems. 
The goal of this project is to be able to correctly identify the type of marine vessel based on the image using Convolutional Neural Networks.


## Methods
The methodology for this project includes data cleaning, exploratory data analysis, feature engineering, and running various models to determine the model 
with the highest accuracy score.

Two types of models will be created including 
Multi Layer Perceptron (MLP) models as well as Convolutional Neural Network (CNN) models. These models will be evalued using accuracy scores to stress the 
importance of minimizing false positives

## Data
The data consisted of a total of 6,252 images from Kaggle. The target variable is broken down into 5 categories.

<img width="591" alt="Screen Shot 2021-07-29 at 4 07 48 PM" src="https://user-images.githubusercontent.com/75099138/127558860-6dcc0e3c-65d0-4208-aa6e-856e316060b1.png">

The images were processed using Keras' image processing tools, including Keras Image Data Generator.

## Exploratory Data Analysis
To become familiar with the data, images from each category in the training set are visualized to see what they look like respectively. some images in the normal train set as well as the pneumonia train set are visualized to see what the images look like. 
Furthermore, a graph is generateed to explore class imbalance within the data.

### Visualization
The images are visualized to compare the different types of classes included in the dataset. The difference are fairly distinct among classes, 
except cargo ships and carrier ships which may be more challenging for the models to distinguish.


**Class 1:Cargo**

<img width="200" alt="Cargo" src="https://user-images.githubusercontent.com/75099138/127561091-f8ad1a38-213b-46b0-ac06-9e22ade77823.png">


**Class 2: Military**

<img width="200" alt="Military" src="https://user-images.githubusercontent.com/75099138/127561143-847016eb-e455-45d6-b5e8-efa02134c465.png">


**Class 3: Carrier**

<img width="200" alt="Carrier" src="https://user-images.githubusercontent.com/75099138/127561159-5acec2d6-ac46-4ff6-af54-1bfac848e6ae.png">

**Class 4: Cruise**

<img width="200" alt="Cruiser" src="https://user-images.githubusercontent.com/75099138/127561186-597df3fe-8572-46c4-9cc4-16d02b53134e.png">


**Class 5: Tanker**

<img width="200" alt="Tanker" src="https://user-images.githubusercontent.com/75099138/127561436-6cf3cf05-23e3-4f3f-9511-21836caba424.png">






### Class Imbalance
In plotting the counts for each of the five classes of ships, we can see a slight class imbalance between the five categories with more of class 1: Cargo, then the rest. To rectify this, a class weight is created to use when modeling.

![class_imbalance](https://user-images.githubusercontent.com/75099138/127192651-9c8b5770-c0e0-4fc7-a31d-4c630a7e9960.png)

## Models
Various models are generated in order to find a model with the best accuracy score. The MLP (MultiLayerPerceptron) models are created first, followed by the CNN (Convultion Neural Net) models. There was a high amount of
overfitting in most models so regularization techniques including l1 regularization, early stopping, and dropout (l2) are implemented in order to minimize overfitting. 
The best model that was did not using transfer learning was the sixth model  which included all three regularization techniques. Overall, the best model was the final model which used the pretrained VGG16 model to train on the data.

## Final Model
The final model that was not pretrained included an l1 of 0.0001, activation of 'relu' and 'softmax', and 0.5 dropout , and early stopping. With this model the accuracy on the train set is 0.818, recall on the validation set is 0.8034, and recall on the test set is 1. 

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
