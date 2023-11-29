# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:27:26 2021

@author: Samiran
"""

"""Using a dataset of of nearly 28,500 credit card transactions and
 multiple unsupervised anomaly detection algorithms, we are going to identify
 transactions with a high probability of being credit card fraud. 

In this project, we will build and deploy the following two ML algorithms:
Local Outlier Factor (LOF)
Isolation Forest Algorithm

Furthermore, using metrics suchs as precision, recall, and F1-scores, 
we will investigate why the classification accuracy for these algorithms 
can be misleading."""



import pandas as pd
data = pd.read_csv('creditcard.csv')
features = data.columns.to_list()
features = [i for i in features if i not in ['Class']]
X = data[features]
Y = data['Class']

"""-----------------------------------------------------------------"""
# Determine number of fraud cases in dataset
Fraud = data [data['Class']==1]
Valid = data [data['Class']==0]

print('no of fraud cases =', len(Fraud))
print('no of valid cases =', len(Valid))

outlier_fraction = len(Fraud)/len(Valid)
print('outlier fraction =',outlier_fraction)

"""-----------------------------------------------------------------"""
#CORRELATION MATRIX
corrmat = data.corr()
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (10,10))
import seaborn as sns
sns.heatmap(corrmat,vmax=0.8,square= True)
plt.show()


"""-----------------------------------------------------------------"""
"""
An outlier is a data point that is different or far from the rest of the data points.

Local outlier factor (LOF) :
It is an algorithm that identifies the outliers present in the dataset.
But what does the local outlier mean?
When a point is considered as an outlier based on its local neighborhood, 
it is a local outlier. LOF will identify an outlier considering the density
of the neighborhood. LOF performs well when the density of the data 
is not the same throughout the dataset. 


Isolation Forest Algorithm for Anomaly Detection:
Anomaly means something which is an unexpected or abnormal event.    
Isolation forest identifies anomalies by isolating outliers in the data.
Isolation forest exists under an unsupervised machine learning algorithm.

Isolation forest works on the principle of the decision tree algorithm. 
It isolates the outliers by randomly selecting a feature from the given 
set of features and then randomly selecting a split value between the 
maximum and minimum values of the selected feature. This random partitioning 
of features will produce smaller paths in trees for the anomalous data values
and distinguish them from the normal set of the data.

Isolation forest works on the principle of recursion. 
This algorithm recursively generates partitions on the datasets by randomly 
selecting a feature and then randomly selecting a split value for the feature.
Arguably, the anomalies need fewer random partitions to be isolated compared 
to the so defined normal data points in the dataset.

Therefore, the anomalies will be the points that have a shorter path in the tree.
Here, we assume the path length is the number of edges travers"ed from the root node.

"""
"""-----------------------------------------------------------------"""
 
#DEFINING OUTLIER DETECTION TOOLS 
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#CREATING A DICTIONARY
classifiers = {'Isolation Forest' : IsolationForest(max_samples=len(X),
                                                   contamination= outlier_fraction,
                                                   random_state=5),
              'Local Outler Factor' : LocalOutlierFactor(n_neighbors=20,
                                                  contamination= outlier_fraction)}



#TRAINING THE MODEL
from sklearn.metrics import classification_report, accuracy_score
plt.Figure(figsize=(10,10))
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
        
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

















 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


