#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised Learning - Classification
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

iris.feature_names

Data_iris = iris.data

Data_iris = pd.DataFrame(Data_iris, columns=iris.feature_names)

Data_iris['label'] = iris.target

plt.scatter(Data_iris.iloc[:,2], Data_iris.iloc[:, 3], c=iris.target)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

x = Data_iris.iloc[:,0:4]
y = Data_iris.iloc[:,4]

"""K-NN Classifier"""

from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=1)

kNN.fit(x,y)

x_N = np.array([[5.6,3.4,1.4,0.1]])

kNN.predict(x_N)

x_N2 = np.array([[7.5, 4, 5.5,2]])

kNN.predict(x_N2)

x_N3 = np.array([[7,4,1.2,1.1]])

kNN.predict(x_N3)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size=0.8,
                                                    random_state=88, shuffle=True, stratify=y)


from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=6, metric= 'minkowski', p=1)

kNN.fit(x_train,y_train)


predicted_types = kNN.predict(x_test)

from sklearn.metrics import accuracy_score 

accuracy_score(y_test, predicted_types)

"""
Decision Tree
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Dt = DecisionTreeClassifier()

Dt.fit(x_train, y_train)

Predicted_types_Dt = Dt.predict(x_test)

accuracy_score(y_test, Predicted_types_Dt)

from sklearn.model_selection import cross_val_score

Scores_Dt = cross_val_score(Dt, x, y, cv=10)

"""
Naive Bayes Classifier
"""

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()

NB.fit(x_train, y_train)

Predicted_types_NB = NB.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, Predicted_types_NB )


from sklearn.model_selection import cross_val_score

Scores_NB = cross_val_score(Dt, x, y, cv=10)

"""
Logistic Regression
"""
from sklearn.datasets import load_breast_cancer

Data_C = load_breast_cancer()

x = Data_C.data
y = Data_C.target
#How Can I create my data and my target?
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,train_size=0.7, random_state=88)

from sklearn.linear_model import LogisticRegression

Lr = LogisticRegression()

Lr.fit(x_train, y_train)

Predicted_classes_Lr = Lr.predict(x_test)

"""
Evaluation Metrics
"""

from sklearn.metrics import confusion_matrix, classification_report

conf_Mat = confusion_matrix(y_test, Predicted_classes_Lr)

Class_rep = classification_report(y_test, Predicted_classes_Lr)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

y_prob = Lr.predict_proba(x_test)

y_prob = y_prob[:,1]

FPR, TPR, Threshoulds = roc_curve(y_test, y_prob)

plt.plot(FPR, TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_prob)

