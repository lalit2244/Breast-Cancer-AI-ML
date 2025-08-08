# Breast-Cancer-AI-ML
This project builds a binary classification model using Logistic Regression. It involves data splitting, standardization, model training, and evaluation with metrics like precision, recall, confusion matrix, and ROC-AUC, along with threshold tuning and sigmoid function understanding.
Project Title:
Binary Classification using Logistic Regression

Objective:
The main goal of this project is to create a machine learning model that can classify data into two categories (yes/no, true/false, 0/1). We will do this using Logistic Regression, a simple but powerful classification algorithm.

What You’ll Learn:
By completing this task, you will learn:

How to work with a binary classification dataset.

How to prepare and split data for training and testing.

How Logistic Regression works and how the sigmoid function makes predictions.

How to measure model performance using metrics like confusion matrix, precision, recall, and ROC-AUC.

How to tune the classification threshold to improve results.

Step-by-Step Process:
Step 1: Choose a Dataset
We start by selecting a binary classification dataset. The Breast Cancer Wisconsin dataset is a good choice because it contains data on tumors labeled as “malignant” (cancer) or “benign” (non-cancer).

Step 2: Import Libraries
We use the following Python libraries:

Pandas – for handling data.

NumPy – for numerical calculations.

Scikit-learn – for machine learning tools like train-test split, logistic regression, and evaluation metrics.

Matplotlib – for visualizing results.

Step 3: Load and Explore Data
We load the dataset and explore it using head() and info() to understand the columns, data types, and whether there are missing values. This step helps us know what we’re working with.

Step 4: Split Data
We divide the data into two parts:

Training set – used to teach the model (usually 70–80% of the data).

Test set – used to check how well the model works on unseen data.

We use train_test_split() from scikit-learn for this.

Step 5: Standardize Features
Logistic Regression works better when features are on a similar scale. We use StandardScaler to transform the data so that each feature has a mean of 0 and standard deviation of 1.

Step 6: Train the Model
We create a Logistic Regression model using LogisticRegression() and train it with the training data using the .fit() method.

Step 7: Make Predictions
We use the .predict() method to get predicted labels for the test set. We can also use .predict_proba() to get the probability scores, which help in threshold tuning.

Step 8: Evaluate the Model
We measure performance using:

Confusion Matrix – shows correct and incorrect predictions.

Precision – out of predicted positives, how many were correct.

Recall – out of actual positives, how many were predicted correctly.

ROC-AUC – measures the overall ability of the model to separate classes.

Step 9: Tune the Threshold
By default, Logistic Regression uses 0.5 as the decision threshold. We can change this value to improve precision or recall depending on the problem’s needs.

Step 10: Explain the Sigmoid Function
The sigmoid function converts any number into a value between 0 and 1. In Logistic Regression, it helps in predicting probabilities.

Conclusion:
This task gives hands-on experience in building a binary classifier, preparing data, applying logistic regression, and evaluating its performance. By the end, you’ll understand how to use this model for real-world problems like spam detection, disease prediction, or fraud detection.
