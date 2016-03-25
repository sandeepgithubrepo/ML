import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


"""
============================================
DO NOT FORGET TO INCLUDE YOUR STUDENT ID
============================================
"""
student_ID = ''

def my_info():
    """
    :return: DO NOT FORGET to include your student ID as a string, this function is used to evaluate your code and results
    """
    return student_ID

#This method creates data from the
#given Normal distribution
def create_samples():
    d1 = np.array([0,0])
    cov_mat1 = np.array([[1,0],[0,1]])
    d2 = np.array([0,0])
    cov_mat2 = np.array([[16,0],[0,16]])
    # generate training samples for y=0 and y=1
    training_data_0 = np.random.multivariate_normal(d1, cov_mat1, 250)
    training_data_1 = np.random.multivariate_normal(d2, cov_mat2, 250)
    #generate testing samples for y = 0 and y = 1
    test_data_0 = np.random.multivariate_normal(d1, cov_mat1, 1000)
    test_data_1 = np.random.multivariate_normal(d2, cov_mat2, 1000)
    
    return training_data_0,training_data_1,test_data_0,test_data_1


#This method applies knn for all given K's
#Also stores misclassification for given train and test data
def apply_knn(k_set,traing_data,train_labels,test_data,test_labels):
    misclass = []
    for k in k_set:
        # p = 2 and metric='minkowski' combination uses euclidean distance 
        knn_train_test = KNeighborsClassifier(n_neighbors=k, weights='uniform',p=2, metric='minkowski',)
        knn_train_test.fit(traing_data, train_labels)
        knn_train_test.predict(test_data)
        acc = knn_train_test.score(test_data,test_labels)
        misclass.append(round(1-acc,3))
            
    return misclass

#This method finds bayes error
def compute_bayes_error():
    np.random.seed(0)
    mu1 = [0, 0]
    cov_mat_1 = 1 * np.eye(2)

    mu2 = [0, 0]
    cov_mat_2 = 16 * np.eye(2)

    #create unified training set from two normal distributions 
    X_vect = np.concatenate([np.random.multivariate_normal(mu1, cov_mat_1, 5000),
                        np.random.multivariate_normal(mu2, cov_mat_2, 5000)])
    y = np.zeros(10000)
    y[5000:] = 1

    # Fit the Naive Bayes' classifier
    clf = GaussianNB()
    clf.fit(X_vect, y)
    # predict the classification probabilities on a grid
    xlim = (-5, 5)
    ylim = (-5, 5)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 70))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)

    acc = clf.score(X_vect,y)
    #Error rate
    error = 1- acc

    #Add decision boundery plot
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle('decision boundary', fontsize=12)
    fig = plt.gcf()
    #set display window title
    fig.canvas.set_window_title('Decision Boundary')
    ax = fig.add_subplot(111)
    p1 = ax.scatter(X_vect[:, 0], X_vect[:, 1], c=y, cmap=plt.get_cmap('Set3'), zorder=5)
    p2 = ax.contour(xx, yy, Z, [0.5],linewidths=3, colors='k')
   
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x1$')
    ax.set_ylabel('$x2$')
    plt.clabel(p2, inline=3, fontsize=5)
    p2.collections[0].set_label("Decision Boundary")
    ax.legend(loc='lower right')
    return error


def draw_plots():
    training_data_0,training_data_1,test_data_0,test_data_1 = create_samples()

    #plot x1->x2 for training data
    training_fig = plt.figure(figsize=(5, 5))
    training_fig.suptitle('training set plot', fontsize=12)
    training_plt = training_fig.add_subplot(111)
    l_yz = training_plt.scatter(training_data_0[:,0],training_data_0[:,1], c= 'red', marker='o')
    l_yo = training_plt.scatter(training_data_1[:,0],training_data_1[:,1], c= 'blue', marker='o')
    training_plt.legend((l_yz, l_yo),
           ('y=0', 'y=1'),
           scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=12)
    training_plt.set_xlabel('$x1$')
    training_plt.set_ylabel('$x2$')
    training_plt = plt.gcf()
    training_plt.canvas.set_window_title('Training set plot')

    #plot x1->x2 for testing data
    testing_fig = plt.figure(figsize=(5, 5))
    testing_fig.suptitle('testing set plot', fontsize=12)
    testing_plt = testing_fig.add_subplot(111)
    y_z = testing_plt.scatter(test_data_0[:,0],test_data_0[:,1], c= 'red', marker='o')
    y_o = testing_plt.scatter(test_data_1[:,0],test_data_1[:,1], c= 'blue', marker='o')
    plt.legend((y_z, y_o),
           ('y=0', 'y=1'),
           scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=12)
    testing_plt.set_xlabel('$x1$')
    testing_plt.set_ylabel('$x2$')
    testing_plt = plt.gcf()
    testing_plt.canvas.set_window_title('Test set plot')
    #making training data and training target values
    train_data = np.concatenate((training_data_0,training_data_1), axis = 0)
    train_labels = np.zeros(500)
    train_labels[250:] = 1

    #making testing data and testing target values
    test_data = np.concatenate((test_data_0,test_data_1), axis = 0)
    test_labels =np.zeros(2000)
    test_labels[1000:] = 1

    k_set = [1,3,5,7,9,13,17,21,25,33,41,49,57]
    
    test_misclass = apply_knn(k_set, train_data, train_labels, test_data, test_labels)
    train_misclass = apply_knn(k_set, train_data, train_labels, train_data, train_labels)

    #Plot decision boundary and Misclassification error rate
    error_rate_fig = plt.figure(figsize=(5, 5))
    error_rate_fig.suptitle('knn misclassification and Bayes error plot', fontsize=12)
    error_rate_plt = error_rate_fig.add_subplot(111)
    error_rate_plt.plot(k_set,train_misclass, 'ro', linestyle='-', label='Train')
    error_rate_plt.plot(k_set,test_misclass, 'bo', linestyle='-', label='Test')
    error_rate_plt.set_autoscaley_on(False)
    error_rate_plt.set_ylim(-0.001)
    error_rate_plt.axhline(y=compute_bayes_error(), linewidth=1, color='k', label='Bayes Error')
    error_rate_plt.legend(loc='lower right')
    error_rate_plt.set_xlabel('$K$')
    error_rate_plt.set_ylabel('$Misclassifications$')
    #show all plots 
    plt.show()

def main():
    #draw the plots for problem-4
    print my_info()
    draw_plots()

if __name__ == '__main__':
    main()



