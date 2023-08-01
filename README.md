# ML_Project_Housing_Prices
ML Assignment for Giza

This file is a project to learn the steps to create successful machine learning models for a given data set.
The data set was obtained from "Kaggle", and has 79 explanatory variables describing aspects of houses in Ames, Iowa region. The aim is to create an ML model that accurately predicts the sale price of a house. The dataset had columns with numerical and categorical data. Many columns had many rows with "NaN" values. 

##Pipeline:
1. Creating a virtual environment thorugh Pyenv and Poetry
2. Downloading Data from Kaggle
3. Visualising data and descriptive statistics using matplotlib and seaborn 
4. Testing different data cleaning techniques such as deleting columns, outliers, scaling etc.
5. Dealing with "NaN" values and categorical data using Ordinal Encoding/ KNN Imputer
6. Applying log transofrmation to change the distribution of data into normal distribution
7. Calculating MI scores to see which features are the most important 
8. Creating new features out of existing ones
9. Splitting data into training and validation sets to test the models accuracy
10. testing 3 ML Algorithms and 1 Neural Network and comapring the R^2 scores

##Results of database exploration: 
After calculating the MI scores, it was interesting to see that the fireplace quality of a house is playing a much more important role than the overall quality of a house when determining the sale price. According to the domain knowledge overall quality is generally much more important. The reason of this score is most probably because many houses don't have a fireplace and therefore have a value of NaN for that column. When treating the NaN values imputation is used and therefore those NaN values received a score. If there is more NaN in the column than other categories the ML algorithm decided that it plays a big role.  

##3 Algorithms:
1. Linear Regression was chosen  since it is one of the simplest algorithms that could be used, where the relationship between the features and the target is linearly modelled. Picking an easier model as the baseline is important to compare other models with it. 

2. XGBoost Regressor is an algorithm that both chooses the "optimal gradient" as well as being efficient. This algorithm is very famous with it's accurate results therefore it has been chosen. GridSearch algorithm was also applied to find the best combination of hyperparameters. 

3. ElasticNet Regression is an algorithm that combines L1 and L2 regularization. It's a balance between the both. L1 regularization helps with feature selection whereas L2 regularization prevents overfitting. They are also both good at handling multicolinearity. Therefore it's expected to fit the dataset. 

##NN:
The neural network chosen had 64-32-16-1 nodes in its layers since increasing the number of nodes results in overfitting. The neural network is also not too deep as the data set relationships are not too complex to be understood by the model. Adam optimizer is used as it is a very common optimizer that obtaines good results. L2 regularization has also been applied to the first two layers of the NN to further reduce effects of high weights and prevent overfitting the model. 

* The best model turned out to be the XGBoost as expected. XGBRegressor algorithm was very successful in this data set since it is designed to predict continuous target variable - Saleprice given the features. It starts by trainig weak decision trees and then makes predictions by combining their outputs in a weighted average. Then learns from mistakes and creates more powerful decision trees. 
* The reason why Linear Regression probably didn't perform well enough is becasue it is highly affected by outliers.   
* The same applied for ElasticNet since it is also a linear regression based algorithm that applies L1 and L2 regualrization to the weights