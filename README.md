

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
Multi Layer Perceptron (MLP) models as well as Convolutional Neural Network (CNN) models. These models will be evaluated using accuracy scores to stress the 
importance of minimizing false positives.

## Data
The data consisted of a total of 6,252 images from Kaggle. The target variable is broken down into 5 categories.

<img width="591" alt="Screen Shot 2021-07-29 at 4 07 48 PM" src="https://user-images.githubusercontent.com/75099138/127558860-6dcc0e3c-65d0-4208-aa6e-856e316060b1.png">

The images were processed using Keras' image processing tools, including Keras Image Data Generator.

## Exploratory Data Analysis
To become familiar with the data, images from each category in the training set are visualized to see what they look like respectively. 
Furthermore, a graph is generateed to explore class imbalance within the data. #See Class Imbalance below

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

**Best Model without Transfer Learning**

<img width="433" alt="Screen Shot 2021-07-29 at 5 18 56 PM" src="https://user-images.githubusercontent.com/75099138/127567233-535cc0a2-0f29-48bb-823b-3e80b84dbb77.png">

The model training accurary is:  93.41

The model validation accurary is:  75.93

**Best Model using Transfer Learning***

<img width="450" alt="Screen Shot 2021-08-04 at 2 40 12 PM" src="https://user-images.githubusercontent.com/75099138/128426048-18babb98-3385-4fbe-a3b9-c90ec6ea6d07.png">

The MobileNetV2 pretrained model training accurary is:  99.1

The MobileNetV2pretrained model validation accurary is:  89.2

<img width="455" alt="Screen Shot 2021-08-04 at 2 41 06 PM" src="https://user-images.githubusercontent.com/75099138/128426201-4e22de97-7d3e-4e86-a1e8-9b8395ce09f0.png">

## Model Predictions: How it Predicts using LIME 

Now lets take a look at some of the model predictions. With Neural Networks its hard to extract feature importance. So, just what does the model see? How does it predict? 

**Example 1: Cargo Ship


<img width="320" alt="Screen Shot 2021-08-05 at 2 33 24 PM" src="https://user-images.githubusercontent.com/75099138/128426502-c06878e8-f923-480c-9bd8-6eb6600f4326.png"><img width="308" alt="Screen Shot 2021-08-05 at 2 36 40 PM" src="https://user-images.githubusercontent.com/75099138/128426515-6e033650-0c74-4063-b386-4c5a43977fa5.png"><img width="322" alt="Screen Shot 2021-08-05 at 2 38 50 PM" src="https://user-images.githubusercontent.com/75099138/128426525-a3e52b20-e3eb-447e-9d30-4ae6ab78ecdd.png"><img width="407" alt="Screen Shot 2021-08-05 at 2 39 56 PM" src="https://user-images.githubusercontent.com/75099138/128426533-d124972b-5420-4df9-94b1-7378d5948651.png">


**Example 2: Cruise Ship


<img width="307" alt="Screen Shot 2021-08-05 at 2 58 30 PM" src="https://user-images.githubusercontent.com/75099138/128426626-14c2a011-7880-4fdd-9035-e9613b3628bf.png"><img width="305" alt="Screen Shot 2021-08-05 at 2 58 39 PM" src="https://user-images.githubusercontent.com/75099138/128426630-e757fc8c-7cf8-444d-8f53-c08b2958ea2b.png"><img width="320" alt="Screen Shot 2021-08-05 at 2 58 45 PM" src="https://user-images.githubusercontent.com/75099138/128426649-73f2206c-b1cb-45ec-86ef-73676e63bc06.png">
<img width="375" alt="Screen Shot 2021-08-05 at 2 58 52 PM" src="https://user-images.githubusercontent.com/75099138/128426658-f89fd81f-b776-4b82-8297-bf9d4d038796.png">





## Summary

 **What worked**


> In trying to improve the accuracy scores, the following seemed to make a difference, most of which is intuitive:

*   Add more layers
*   Adding stronger regularization (though not a huge improvement)
*   Using a pretrained model with established weights (VGG16)
*   Using a pretrained model on transformed images

 **What didnt work**

*   Training models for longer
*   But also not training models for long enough on transformed images.
*   Regularizing (when the dropout layers were not strong enough)


## Next Steps: 
Some future steps to improve the model include:
  - Experiment with other transfer models (ResNet, Xception)
  - Trying other techniques to minimize overfitting, such as batch normalization
  - App Deployment!

## Repository Structure
```
├── best_models
├── data
├── images
├── Final_Notebook.ipynb
├── Presentation.pdf
└── README.md
