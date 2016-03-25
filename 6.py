import numpy as np
from scipy.spatial import distance

import mnist_load_show as mnist

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


def kmeans(X, centriods):
    done = True
    while done:
        distances = distance.cdist(X, centriods, 'sqeuclidean')
        K = np.argmin(distances, axis=1)
        cluster_mean = np.array([ X[K == i].mean(axis=0) for i in range(len(centriods)) ])
        done = not (cluster_mean == centriods).all()
        centriods = np.copy(cluster_mean)

    c = np.array([ X[K == i] for i in range(len(centriods)) ])
    return np.array(cluster_mean), c

def kmedoids(dist, mediod_idx):

    done = True
    mediod_idx = np.array(mediod_idx)
    while done:
        medoids = np.array([ dist[mediod_index] for mediod_index in mediod_idx]).transpose()
        cluster_idx = np.argmin(medoids, axis=1)
        updated_idx = np.zeros(mediod_idx.shape)

        for i in range(len(mediod_idx)):
            curr_idx = np.argwhere(cluster_idx == i).flatten()
            dist_cur_cluster = dist[curr_idx].transpose()[curr_idx]
            cur_mediods = np.argmin(np.sum(dist_cur_cluster, axis=1))
            updated_mediod = curr_idx[cur_mediods]
            updated_idx[i] = updated_mediod

        done = (updated_idx != mediod_idx).any()
        mediod_idx = np.copy(updated_idx)

    return updated_idx, cluster_idx



def main():
    """
    DO NOT TOUCH THIS FUNCTION. IT IS USED FOR COMPUTER EVALUATION OF YOUR CODE
    """
    results = my_info() + '\t\t'
    print results + '\t\t'
    X, Y = mnist.read_mnist_training_data(500)
    centriods = X[:10]
    cm, c = kmeans(X, centriods)
    mnist.visualize(cm)
    #for mean, cluster in zip(cm, c):
        #mnist.visualize(np.insert(cluster, 0, mean, axis=0))


    centriods_unique = np.array([ X[np.where(Y == i)[0][0]] for i in range(10) ])
    cm, c = kmeans(X, centriods_unique)
    mnist.visualize(cm)
    #for mean, cluster in zip(cm, c):
        #mnist.visualize(np.insert(cluster, 0, mean, axis=0))

    distances = distance.cdist(X, X, 'euclidean')
    medoids_idx, clusters = kmedoids(distances, list(range(10)))
    medoids = np.array([X[int(i)] for i in medoids_idx])
    c = np.array([ X[clusters == i] for i in range(10) ])
    mnist.visualize(medoids)
    #for mean, cluster in zip(cm, c):
        #mnist.visualize(np.insert(cluster, 0, mean, axis=0))


    mediod_idx = [ np.where(Y == i)[0][0] for i in range(10) ]
    medoids_idx, clusters = kmedoids(distances, mediod_idx)
    medoids = np.array([X[int(i)] for i in medoids_idx])
    c = np.array([ X[clusters == i] for i in range(10) ])
    mnist.visualize(medoids)
    #for mean, cluster in zip(cm, c):
        #mnist.visualize(np.insert(cluster, 0, mean, axis=0))

    
if __name__ == '__main__':
    main()

