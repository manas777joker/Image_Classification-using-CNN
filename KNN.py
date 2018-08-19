from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split 
from sklearn import metrics 
import numpy as np


def Knn():

	x = np.load('info.npy')
	y = np.load('lb.npy')

	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.025)
	print len(x_train)
	print len(x_test)

	knn = KNeighborsClassifier()
	knn.fit(x_train,y_train)
	pred = knn.predict(x_test)
	#print pred
	#correct = 0
	return (metrics.accuracy_score(y_test,pred)*100)
	#print("Classification report for classifier %s:\n%s\n"
    #	  % (knn, metrics.classification_report(y_test, pred)))
	#print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, pred))


#if __name__ == "__main__":
#	main()