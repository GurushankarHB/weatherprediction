
# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

import numpy as np

seed = 7
np.random.seed(seed)
data = np.loadtxt("data.txt")
  
 
# X -> features, y -> label 
X_train = data[:4646,:12]
Y_train = data[:4646,12:13]
X_test = data[4646:6936,:12]
Y_test = data[4646:6936,12:13]
  
# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, Y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, Y_test) 
print("accuracy =", accuracy);
  
# creating a confusion matrix 
cm = confusion_matrix(Y_test, svm_predictions) 

