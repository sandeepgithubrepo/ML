from __future__ import division
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import resize
from sklearn.metrics import mean_squared_error
from math import sqrt

#Change font size of legend in all plots
#This is global setting 
plt.rc('legend',**{'fontsize':6})

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


#define range
a = -3
b = 3
# define number of samples to be generated 
n = 30
# define mean and standard deviation for the noise data  
mu = 0
sd = 0.4

# generate samples of x within specified range 
x = np.random.uniform(-3,3,n)
x = np.asarray(x)

#compute y as a function of x
#f(x) = 2+x-0.5^x2
def f(xval):
   yval = 2 + xval-0.5*xval*xval
   return yval

# noise data iid 
devs = np.random.normal(mu,sd,n)
    
# create data set
def generate_data():
    #space equally from minimum to max values
    xx = np.sort(x)#np.linspace(min(x), max(x),n)
    y = f(xx)
    y = np.asarray(y)
    ydata = y + devs
    ydata = np.asarray(ydata)
    return xx,ydata
    
# for each degree predict y and 
#compute  R^2
def compute_coeff_determination(xx,ydata,ybar):
    for i in xrange(0,11):
        fig1 = plt.subplot(4,3,i)
        plt.tight_layout()
        fit = np.polyfit(xx,ydata,i)
        y_predict = np.polyval(fit,xx)      
        ssreg = np.sum((y_predict-ydata)**2)  
        sstot = np.sum((ydata-ybar)**2) 
        rsqr  = ssreg / float(sstot)
        r = 1-rsqr
        print "R^2 for degree = ",i,"is = ",r
        fig1.plot(xx,ydata,'o',label="data set")
        fig1.plot(xx,y_predict,'r',label = "predicted curve")
        fig1.plot(xx,y_predict,'o',c = 'r',label = "predicted pts")
        fig1.set_title("degree %d" % i)
        fig1.legend(loc = "lower right")
       

# create validation data set
def create_validation_sets():
    validation_set = np.arange(0,n)
    validation_folds = 10
    #divide array into 10 equal splits
    s = np.asarray(np.split(validation_set,validation_folds))
    return s
    


#perform k-fold validation for the 
#created data set
def compute_sqrd_error(s,xx):   
    sqrd_error = []
    for i in xrange(0,11):
        sqrd_error_sum = 0
        for j in xrange(0,10):
            l = np.arange(0,10)
            #uses leave one out strategy in validation
            l = np.delete(l,j)
            p = np.concatenate(([s[k] for k in l]),axis=0)
            x_fold = [ xx[k] for k in p]
            x_fold = np.asarray(x_fold)
            yd = f(x_fold)
            #fit polynomial for each degree i
            fit = np.polyfit(x_fold,yd,i)
            #predict y for each degree 
            yk_predict = np.polyval(fit,x_fold)
            #compute mean squared error
            mse = mean_squared_error(yd, yk_predict)
            #sum mse up for each split inorder to find mse for each degree  
            sqrd_error_sum = sqrd_error_sum + mse

        sqrd_error.append(sqrd_error_sum)

    return sqrd_error
#plots K Vs sqrd-error-sum
def plot_K_sqrd_error(sqrd_error):

    k_flds = np.arange(0,11)
    sqrd_error = np.asarray(sqrd_error)
    k = np.argmin(sqrd_error)
    #print sqrd_error
    #print "Min K = ",k

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(k_flds,sqrd_error,'o',linestyle='-')
    plt.yscale('log')
    ax.set_xlabel('$K$')
    ax.set_ylabel('$sum of squared error$')
  
    
def polynomial_regression():
    #generate data
    xx,ydata = generate_data()    
    # compute mean of the samples of y
    ybar = np.sum(ydata)/len(ydata)
    #compute R^2 and plot
    compute_coeff_determination(xx,ydata,ybar)
    # create data splits for validation
    s = create_validation_sets()
    #compute squared errors
    sqrd_error = compute_sqrd_error(s,xx)
    # plot K Vs squared errors and 
    plot_K_sqrd_error(sqrd_error)
    plt.show()
    
def main():
    #polynomial  regression  problem-4
    print my_info()
    polynomial_regression()

if __name__ == '__main__':
    main()
