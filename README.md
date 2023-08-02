# ML_Project_Housing_Prices
*ML Assignment for Gizatech*

## Introduction
This project aims to create successful machine learning models to predict the sale prices of houses in the Ames, Iowa region. The dataset, obtained from "Kaggle", contains 79 explanatory variables describing various aspects of the houses. The project involves data cleaning, preprocessing, feature engineering, and the implementation of multiple algorithms to achieve accurate predictions.

## Contents
* This project consists of a **dataset folder** which includes all data taken from Kaggle (not shown in GitHub). 
* There is a **Poetry_VR_Environment folder** which holds the whole project files in it.
* Inside the Poetry_VR_Environment folder one can find the main project file, **Predicting_Housing_Prices**, and the 3 other projects as **project2**, **project3**, **project4** as .ipynb files. 
* It also includes a **function_sheet** as a .py file which contains importable functions used throughout the projects.
* An **output.txt** file is output of a code block (not shown in GitHub).
* **.gitignore** to hide folders/files from being visible in GitHub.
* A **poetry.lock** and a **pyproject.toml** file is created thorugh Poetry + Pyenv and are used to manage dependencies and versions in the project.

## Setup and Installation 
To run the project, follow these steps:
1. Install Pyenv and Poetry.
2. Clone the repository and navigate to the Poetry_VR_Environment folder.
3. Create a virtual environment: pyenv virtualenv 3.9.0 ml_project_env.
4. Activate the environment: pyenv activate ml_project_env.
5. Install project dependencies: poetry install. 

## Dataset description
The dataset contains a mix of numerical and categorical data with 19 columns containing NaN values.
![Histogram](https://user-images.githubusercontent.com/123895232/257803209-5c474aec-993b-4920-9311-58e4a3680a49.png)
Above are the produced histograms for numerical columns in the data to better visualize the distributions.
## Pipeline of the main project
1. Creating a virtual environment thorugh Pyenv and Poetry
2. Downloading Data from Kaggle
3. Visualising data and descriptive statistics using matplotlib and seaborn 
4. Testing different data cleaning techniques such as deleting columns, scaling etc.
5. Dealing with "NaN" values and categorical data using Ordinal Encoding/ KNN Imputer
6. Applying log transofrmation to change the distribution of data into normal distribution
7. Calculating MI scores to see which features are the most important 
8. Creating new features out of existing ones
9. Splitting data into training and validation sets to test the models accuracy
10. Testing 3 ML Algorithms and 1 Neural Network and comapring the R^2 scores

## Results of database exploration 
After calculating the MI scores, it was interesting to see that the fireplace quality of a house is playing a much more important role than the overall quality of a house when determining the sale price. According to the domain knowledge overall quality is generally much more important. The reason of this score is most probably because many houses (690 rows) don't have a fireplace and therefore have a value of NaN for that column. When treating the NaN values imputation is used and therefore those NaN values received a score. The number of NaN valued rows is higher than any other category row. Therefore the ML algorithm decided that the "NaN" value which is replaced by a certain number plays a big role.  
## 3 Algorithms
1. Linear Regression was chosen  since it is one of the simplest algorithms that could be used, where the relationship between the features and the target is linearly modelled. Picking an easier model as the baseline is important to compare other models with it. 

2. XGBoost Regressor is an algorithm that both chooses the "optimal gradient" as well as being efficient. This algorithm is very famous with it's accurate results therefore it has been chosen. GridSearch algorithm was also applied to find the best combination of hyperparameters. 

3. ElasticNet Regression is an algorithm that combines L1 and L2 regularization. It's a balance between the both. L1 regularization helps with feature selection whereas L2 regularization prevents overfitting. They are also both good at handling multicolinearity. Therefore it's expected to fit the dataset. 

## NN
The neural network chosen had 64-32-16-1 nodes in its layers since increasing the number of nodes results in overfitting. The neural network is also not too deep as the data set relationships are not too complex to be understood by the model. Adam optimizer is used as it is a very common optimizer that obtaines good results. L2 regularization has also been applied to the first two layers of the NN to further reduce effects of high weights and prevent overfitting the model. 
Later on this approach was found to be too complex for this little data and the NN was reduced to a 16-12-1. The regularization parameters were also removed. The result turned out to be better for training loss. However the validation loss was higher for the second model.   
![Loss_Graph_Model1](https://user-images.githubusercontent.com/123895232/257804546-6b02268a-fd24-4746-ab14-ca83d789494d.png)
![Loss_Graph_Model2](https://user-images.githubusercontent.com/123895232/257804936-3cd8c01a-3338-4c08-94cc-91511faf9148.png)

* The best model turned out to be the XGBoost as expected. XGBRegressor algorithm was very successful in this data set since it is designed to predict continuous target variable - Saleprice given the features. It starts by trainig weak decision trees and then makes predictions by combining their outputs in a weighted average. Then learns from mistakes and creates more powerful decision trees. 
* The reason why Linear Regression probably didn't perform well enough is becasue it is highly affected by outliers.   
* The same applied for ElasticNet since it is also a linear regression based algorithm that applies L1 and L2 regualrization to the weights

------------

After the XGBoost was chosen as the algorithm for this dataset, it was decided to test out whether different approaches in the data cleaning would result in more accurate predictions by the XGBoost algorithm and therefore 3 new side projects were created.

## Project 2
Steps: 
1. Download data set
2. Make ID column the name of the rows
3. Delete columns with NaN data present
4. Apply Ordinal Encoding to deal with categorical data
5. Apply log transformation to turn the data into normal distribution
6. Calculate MI scores
7. Create new features
8. Split the data into training and validation sets
9. Apply scaling
10. Apply XGBoost algorithm
11. Apply GridSearch algorithm to tune the hyperparameters

Project 2 did not perform better than the original project. 
- It had a final R^2 score of 0.908

## Project 3
Steps: 
1. Download data set
2. Make ID column the name of the rows
3. Delete columns where NaN data is present in more than half of the rows
4. Apply Ordinal Encoding to deal with categorical data
5. Apply KNN imputation to deal with remaining rows where NaN data is still present
6. Calculate MI scores
7. Create new features
8. Delete columns that make up the new features
9. Delete columns with least MI scores
10. Split the data into training and validation sets
11. Apply scaling
12. Apply XGBoost algorithm
13. Apply GridSearch algorithm to tune the hyperparameters

Project 3 did not perform better than the original project. However, it was better than Project 2. 
- It had a final R^2 score of 0.919

## Project 4 
*NOTE: The main idea of filling NaN values in different columns with different values was taken from another Kaggle participant who dealt with the same dataset. It was included to compare existing model perfromances with somthing from outside.*
Steps: 
1. Download data set
2. Make ID column the name of the rows
3. Fill NaN values using different methods
4. Apply One-Hot Encoding to deal with nominal data
5. Calculate MI scores
6. Create new features
7. Split the data into training and validation sets
8. Apply scaling
9. Apply XGBoost algorithm
10. Apply GridSearch algorithm to tune the hyperparameters  

Project 3 did not perform better than any of the projects. 
- It had a final R^2 score of 0.907

## Results and Comparison
All models with different data cleaning methods had similar R^2 scores. However the initial approach turned out to be the best one with an R^2 score of 0.929. The reason might be the removal of certain features from the dataset which were suspected to be low-impact. This might have helped the algorithm to learn faster and more accurately since it didn't need to consider certain features with low MI scores which suggests that knowing those does not make one more confident about the target. Best 2 projects applied this method and received a gerater R^2 score.
![Regression_Graph_Project1](https://user-images.githubusercontent.com/123895232/257839789-f78d9bc1-5aa0-4c78-979b-8db0694a4718.png)

![Regression_Graph_Project2](https://user-images.githubusercontent.com/123895232/257839382-3963891c-ed06-4ffd-9cc4-2855d157b534.png)

![Regression_Graph_Project3](https://user-images.githubusercontent.com/123895232/257810370-078847aa-2bbe-455b-93b5-a7f52a33a5f0.png)

![Regression_Graph_Project_4](https://user-images.githubusercontent.com/123895232/257810561-4715cb08-d473-4436-a96d-165fb936ea53.png)

Above are regression graphs created for each XGBoost model prediction and the actual price. Closer the points are to the best-fit line (red line) more accurate the model is. As it can be seen in the first graph the points are well aligned around the red line with little fluctuations compared to other projects thus prooving it is the best model out of all 4.  