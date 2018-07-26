from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split 
from sklearn import metrics 
import numpy as np

x = np.load('X.npy')
y = np.load('Y.npy')

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.025)
print len(x_train)
print len(x_test)
'''
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
#print pred
correct = 0
total=len(y_test)
for i in range(0,total) :
    if(y_test[i] == pred[i]):
        correct+=1
print float(correct)/total*100
'''
'''
Ml = MLPClassifier(activation='relu',alpha=1e-05,momentum=0.9)
Ml.fit(x_train,y_train)
pred = Ml.predict(x_test)
correct=0
total = len(y_test)
for i in range(0,total):
    if(y_test[i]==pred[i]):
        correct+=1
#print correct
print float(correct)/total*100

print("Classification report for classifier %s:\n%s\n"
      % (Ml, metrics.classification_report(y_test, pred)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, pred))
'''
print '-------------SVM------------------'

sv = SVC(max_iter=100)
sv.fit(x_train,y_train)
predict = sv.predict(x_test)
print(metrics.accuracy_score(y_test,predict)*100)
print("Classification report for classifier %s:\n%s\n"
      % (sv, metrics.classification_report(y_test, predict)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predict))