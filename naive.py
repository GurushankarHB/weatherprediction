from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
import numpy as np

seed = 7
np.random.seed(seed)
data = np.loadtxt("data.txt")
  
 
# X -> features, y -> label 
X_train = data[:4646,:12]
Y_train = data[:4646,12:13]
X_test = data[4646:6936,:12]
Y_test = data[4646:6936,12:13]
  
# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, Y_train) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test 
accuracy = gnb.score(X_test, Y_test) 
print("accuracy =", accuracy);
  
# creating a confusion matrix 
cm = confusion_matrix(Y_test, gnb_predictions) 
