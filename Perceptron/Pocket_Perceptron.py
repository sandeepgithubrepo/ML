import numpy as np
import mnist_load_show as mnist
from sklearn.metrics import confusion_matrix

__author__ = 'Sandeep Panchamukhi'

#load training data
X, y = mnist.read_mnist_training_data()

#training/test data size 
size = 30000
#create training and test data sets
Xtrain = X[0:size]
ytrain = y[0:size]
Xtest = X[size : size + size]
ytest = y[size: size + size]

#feature vectors of each data point 
feat_count = len(Xtrain[0])

"""
This method computes error rate for the
given confusion matrix
"""
def error_rate(cm):
    tp = cm.trace()
    fp = cm.sum()-tp
    acc = tp/float((tp+fp))
    return 1-acc

""" 
implements Perceptron with Pocket to hold 
the best seen weight vector by storing number
of updates and history of weight vector
cur_label parameter tells for which digit
training is carrying out 
"""

def pocket_perceptron(xtrain, ytrain, cur_label):
    
    feat_count = len(xtrain[0])
    #weight vector
    w_vec = np.zeros(feat_count)
    score = 0
    #flag to hold solution achieved or not
    converge = False
    score_p = 0
    w_p = w_vec
    #max iterations in case solution not converged
    epochs = 20
    samples,f_count = xtrain.shape
    # one digit labels with +1 and -1 for rest of the digits
    y_expected = [ 1 if y == cur_label else -1 for y in ytrain ]
    
    for epoch in xrange(epochs):  
            #mark as converged        
            converge = True
            for i in xrange(0,samples):
                x = xtrain[i]     
                y_pred = np.sign(np.dot(x, w_vec))
                if y_expected[i] != y_pred:
                    #update best score and weight history vector
                    if score > score_p:
                        score_p = score
                        w_p = w_vec
                        #labels do not match and hence solution not yet 
                        #converged
                        converge = False
                    #update current weight vector
                    w_vec = w_vec + y_expected[i]*x
                    #reset update counter in case label mismatch
                    score = 0
                else:
                    #labels matched increment counter
                    score += 1
            #stop iteration if solution is converged        
            if converge:
                break   
    return w_p

"""
This method trains classifier by 
one versus all technique
calls pocket_perceptron for each digit
"""

def ova_perceptron_train(Xtrain,ytrain):
    w_ova = [ pocket_perceptron(Xtrain, ytrain, i) for i in xrange(0,10) ]
    w_mat_ova = np.matrix(w_ova)
    return w_mat_ova

"""
predicts label by one versus all technique
using x_mat and weight vector 
"""

def predict_ova(x_mat, wt_vec):
    x_mat_trans = x_mat.transpose()
    pred = np.argmax(wt_vec.dot(x_mat_trans))
    return pred

"""
This method is used to classify all
test data points using one versus all technique
calls predict_ova for each test data point
computes confusion matrix using yactual and ypredicted labels 
for test data
"""
def ova_perceptron_test(xtest,ytest,w_mat_ova):
    ytest_pred = [predict_ova(x, w_mat_ova) for x in xtest]
    cm_ova = confusion_matrix(ytest, ytest_pred)
    return  cm_ova


"""
This method trains the classifier by
all versus all technique
we create 45 classifiers by checking i < j
fills weight vector[j[i] using weight vector[i[j] value
"""
def ava_perceptron_train(xtrain,ytrain):
    w_ava = [ np.zeros((10,feat_count)) for i in xrange(0,10) ]
    for i in xrange(10):
        for j in xrange(10):
            if (i != j) and i < j : 
                #create smaller data sets
                i_j_train = np.logical_or(ytrain ==i ,ytrain == j)
                x_i_j_train = xtrain[i_j_train]
                y_i_j_train = ytrain[i_j_train]
                #train pocket perceptron
                w_ava[i][j] = pocket_perceptron(x_i_j_train, y_i_j_train, i)
                #fill w_ava[j][i] using w_ava[i][j] value
                w_ava[j][i] = np.subtract(1,w_ava[i][j])
    return w_ava

"""
predict the label of the test data
by using all versus all technique
"""
def predict_ava(x_mat, w_vec):
    x_mat_trans = x_mat.transpose()
    w_x_prod_sum = [ np.sum(np.sign(w.dot(x_mat_trans))) for w in w_vec]
    pred = np.argmax(w_x_prod_sum)
    return pred


"""
This method classify the test data using
all versus all technique
computes confusion matrix for all versus all technique
"""

def ava_perceptron_test(xtest,ytest,w_mat_ava):
    ytest_pred = [ predict_ava(np.matrix(x), w_mat_ava) for x in xtest]
    ytest_pred = np.asarray(ytest_pred)
    cm_ava = confusion_matrix(ytest, ytest_pred)
    return cm_ava



def one_vs_all():
    """
    Implement the the multi label classifier using one_vs_all paradigm and return the confusion matrix
    :return: the confusion matrix regarding the result obtained using the classifier
    """
    w_mat_ova = ova_perceptron_train(Xtrain,ytrain)
    one_vs_all_conf_matrix = ova_perceptron_test(Xtest,ytest,w_mat_ova)
    return one_vs_all_conf_matrix


def all_vs_all():
    """
    Implement the multi label classifier based on the all_vs_all paradigm and return the confusion matrix
    :return: the confusing matrix obtained regarding the result obtained using teh classifier
    """
    w_mat_ava = ava_perceptron_train(Xtrain,ytrain)
    all_vs_all_conf_matrix = ava_perceptron_test(Xtest, ytest, w_mat_ava)
    return all_vs_all_conf_matrix




def main():
    """
    DO NOT TOUCH THIS FUNCTION. IT IS USED FOR COMPUTER EVALUATION OF YOUR CODE
    """
    
    results = np.array_str(np.diagonal(one_vs_all())) + '\t\t'
    results += np.array_str(np.diagonal(all_vs_all()))
    print results + '\t\t'

if __name__ == '__main__':
    main()

