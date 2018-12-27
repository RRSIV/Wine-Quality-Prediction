# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 19:41:55 2018

@author: Siva
"""
##Importing All Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score

##Importing the dataset
wine_red = pd.read_csv('winequality-red.csv',sep = ';')
figure = plt.figure(figsize = (14,10))
##Using seaborn library to visualize correlations if any
corr = wine_red.corr()
sns.heatmap(corr,center=0,
            square=True, linewidths=.8)

##Creating bins for wine quality (2-5 Bad and 5-8 as Good)
bins = [2,6,8]
names = ['Bad','Good']
wine_red['quality'] = pd.cut(wine_red['quality'],bins = bins, labels = names)

##using label encoder to encode values of bad as 0 and good as 1
label_quality = LabelEncoder()
wine_red['quality'] = label_quality.fit_transform(wine_red['quality'])
## Counting the number of good and bad wines
sns.countplot(wine_red['quality'])
##Assigning 
X = wine_red.drop('quality', axis = 1)
y = wine_red.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling to put all variables on the same scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


##Applying Random Forest algorithm to see results with 200 as number of estimaters

random_forest = RandomForestClassifier(n_estimators=200)
##Fitting the model to the training set
random_forest .fit(X_train, y_train)
##Predicting the value of test set using the trained model
y_pred_rf= random_forest.predict(X_test)
##Confusion Matrix to see the accuracy of the model and it turns out to be 92.5%
confusion_matrix = confusion_matrix(y_test, y_pred_rf)

y_pred_rf
print(confusion_matrix)
##Using alterante Model(SVM) to see the model performance
svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
##Confusion Matrix for the SVM mmodel with acuuracy 91.5%
cm_svc=confusion_matrix(y_test, y_pred_svc)
print(cm_svc)
91
##Performing 10 fold cross validation to increase the performance of the Random Forest, turns out that thereis no much improvement
rfc_eval = cross_val_score(estimator = random_forest, X = X_train, y = y_train, cv = 35)
rfc_eval.mean()



