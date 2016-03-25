from __future__ import division
import numpy as np
import mnist_load_show as mnist
from sklearn.metrics import confusion_matrix

'''
use pdis in order to find the the distance
'''
from scipy.spatial.distance import cdist
import numpy as np
"""
============================================
DO NOT FORGET TO INCLUDE YOUR STUDENT ID
============================================
"""
student_ID = ''

#load data
X, y = mnist.read_mnist_training_data()
#divide training and test data each of size 2500 images
Xtrain,ytrain = X[0:2500],y[0:2500]
Xtest,ytest = X[2500:5000],y[2500:5000]

#This method  computes mean of the images
def compute_average(imlist):
   """  Compute the average of a list of images. """
   mean_matrix = np.mean(np.array([ m for m in imlist]), axis=0 )
   return mean_matrix

# This method computes error rate
# of the classifer based on the input
# confusion matrix
def error_rate(cm):
     tp = cm.trace()
     fp = cm.sum()-tp
     acc = tp/(tp+fp)
     return 1-acc



def my_info():
    """
    :return: DO NOT FORGET to include your student ID as a string, this function is used to evaluate your code and results
    """
    return student_ID


def KNN():
    """
    Implement the classifier using KNN and return the confusion matrix
    :return: the confusion matrix regarding the result obtained using knn method
    """
    
    # Computes distance between each pair of the two collections of inputs.
    pair_distace_collection = cdist(Xtest,Xtrain,'euclidean')

    # dimensions of the pairwise dist array
    r,c = pair_distace_collection.shape

    # classify the test image into the class for
    # which the predicted class is the class of the
    # closest training image
    # argmin gives the index of the smallest element
    # use this index to get corresponding prediction from the training label 
    ypred = [ ytrain[np.argmin(pair_distace_collection[i])] for i in xrange(0,r)]
   
    # compute confusion matrix
    knn_conf_matrix = confusion_matrix(ytest, ypred)
    #print "Error Rate",error_rate(knn_conf_matrix)
    return knn_conf_matrix


def simple_EC_classifier():
    """
    Implement the classifier based on the Euclidean distance
    :return: the confusing matrix obtained regarding the result obtained using simple Euclidean distance method
    """
    avg_digits = []
    for i in xrange(0,10):
        # for each prototype class labels get training samples
	class_train = ytrain[:,] == i
        # training samples corresponding to the selected label
	clzt  = Xtrain[class_train]
        # compute mean of all samples belonging to the same class
	mean_matrix = compute_average(clzt)
        # store mean digit of each class
        avg_digits.append(mean_matrix)

    #convert to numpy array
    mean_digits = np.asarray(avg_digits)
  
    # Computes distance between each pair of the two collections of inputs.
    pair_distace_collection = cdist(Xtest,mean_digits,'euclidean')

    # dimesions of pairwise distance array
    r,c = pair_distace_collection.shape
    
    # classify the test image into the class 
    # for which the distance to the prototype is the smallest
    # argmin gives the index of the smallest element in the array
    ypred = [np.argmin(pair_distace_collection[i]) for i in xrange(0,r)]

    # compute confusion matrix
    simple_EC_conf_martix = confusion_matrix(ytest, ypred)

    #Print Error rate
    #print "Error Rate",error_rate(simple_EC_conf_martix)
    return simple_EC_conf_martix




def main():
    """
    DO NOT TOUCH THIS FUNCTION. IT IS USED FOR COMPUTER EVALUATION OF YOUR CODE
    """
    results = my_info() + '\t\t'
    results += np.array_str(np.diagonal(simple_EC_classifier())) + '\t\t'
    results += np.array_str(np.diagonal(KNN()))
    print results + '\n'

if __name__ == '__main__':
    main()
