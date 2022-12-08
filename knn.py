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
  
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, Y_train) 
  
# accuracy on X_test 
accuracy = knn.score(X_test, Y_test) 
print("accuracy =", accuracy); 
  
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(Y_test, knn_predictions) 


