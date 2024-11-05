# M2-Deliverable

This project uses machine learning algorithms such as Logistic Regression, KNN + KMeans and SVM, predict whether or not an individual has heart disease.
Metrics like accuracy, precision and F1 Score, were used to evaluate the model's performance 

URL : https://github.com/Lori-A/M2-Deliverable

The project includes:

Data cleaning and preprocessing.
Feature Engineering (StandardScaler).
Logistic Regression Model
KNearest Neighbours and KMeans
Prediction of Heart Disease(Binary; 0 - 'No , 1 - 'Yes'
Visualization of prediction.
A graph comparing CPU times
Accuracy and F1 score for evaluation

Dependencies:
To run this project, you need to install the following Python packages:

Python 3.x
pandas
numpy
scikit-learn: from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score
import pandas as pd
import time


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.neighbors._classification")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler 
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, r2_score, f1_score
from sklearn.preprocessing import StandardScaler 
import time
import pandas as pd
import matplotlib.pyplot as plt




IDE : Jupyter Notebook
