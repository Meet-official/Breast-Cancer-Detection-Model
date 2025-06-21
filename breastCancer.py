# Breast Cancer Detection Model

# Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection & Data Processing

## Importing the dataset from drive
dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Projects/Breast_Cancer_Classification/data.csv')

dataset.drop(['Unnamed: 32','id'],axis=1,inplace=True)

## Loading the data to a dataframe
data_frame = pd.DataFrame(dataset)

## Number of rows and columns
data_frame.shape

## Info of data
data_frame.info()

## Checking for the missing values
data_frame.isnull().sum()

## Statistical measures about the data
data_frame.describe()

## Converting alphabetical data to numeric
data_frame['diagnosis'] = data_frame['diagnosis'].map({'M':1,'B':0})

## Droping last 10 rows for random data prediction (not as test case)
forPrediction = data_frame.tail(10)
data_frame.drop(data_frame.tail(10).index,inplace=True)

## Checking the distribution of 'diagnosis' variable
data_frame['diagnosis'].value_counts()

"""
Here, in diagnosis:
*   0 -> Benign Tumor
*   1 -> Malignant Tumor
"""

# Groupby diagnosis and getting mean of all, From this we can observe the data of benign cases and malignant cases
data_frame.groupby('diagnosis').mean()
# From this, We can observe that the values of 'Malignant' cases are higher as compared to the values of 'Benign' cases.


# Splitting the data into Training and Testing
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data_frame, data_frame['diagnosis']):
    train = data_frame.loc[train_index]
    test = data_frame.loc[test_index]

print(data_frame.shape, train.shape, test.shape)


# Separating the Features and the Target

X_train = train.drop(columns='diagnosis', axis=1)  # Features
Y_train = train['diagnosis']   # Target

X_test = test.drop(columns='diagnosis', axis=1)  # Features
Y_test = test['diagnosis']   # Target


# Model Training

## Using Logistic Regression:
model = LogisticRegression()

## Training the model using training data
model.fit(X_train, Y_train)


# Model Evaluation

## Accuracy Score

### Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print("Accuracy on training data: ", training_data_accuracy)


### Accuracy on testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Accuracy on test data: ", test_data_accuracy)


# Building a Predictive System

input_data = (20.6,29.33,140.1,1265,0.1178,0.277,0.3514,0.152,0.2397,0.07016,0.726,1.595,5.772,86.22,0.006522,0.06158,0.07117,0.01664,0.02324,0.006185,25.74,39.42,184.6,1821,0.165,0.8681,0.9387,0.265,0.4087,0.124)

## Changing the input data to a numpy array
idnpary = np.asarray(input_data)

## Reshaping the numpy array as we are predicting for one data point
input_data_reshaped = idnpary.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if (prediction[0] == 0):
  print("The Breast Cancer is Benign")
else:
  print("The Breast Cancer is Malignant")