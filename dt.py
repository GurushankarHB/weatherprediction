
# importing necessary libraries 
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
  
# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, Y_train) 
dtree_predictions = dtree_model.predict(X_test)

# accuracy on X_test 
accuracy = dtree_model.score(X_test, Y_test) 
print("accuracy =", accuracy); 
  
# creating a confusion matrix 
cm = confusion_matrix(Y_test, dtree_predictions) 

