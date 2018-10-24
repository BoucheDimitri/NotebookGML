import numpy as np
import matplotlib.pyplot as pyplot
import scipy.spatial.distance as sd
import sklearn.cluster as skc
import sklearn.metrics as skm
import scipy
import scipy
import sys
import os

path=os.path.dirname(os.getcwd())
sys.path.append(path)
from helper import *
from graph_construction.func import *











def build_laplacian(W, laplacian_normalization=""):
#  laplacian_normalization:
#      string selecting which version of the laplacian matrix to construct
#      either 'unn'normalized, 'sym'metric normalization
#      or 'rw' random-walk normalization

#################################################################
# build the laplacian                                           #
# L: (n x n) dimensional matrix representing                    #
#    the Laplacian of the graph                                 #
#################################################################


#################################################################
#################################################################









def spectral_clustering(L, chosen_eig_indices, num_classes=2):
#  Input
#  L:
#      Graph Laplacian (standard or normalized)
#  chosen_eig_indices:
#      indices of eigenvectors to use for clustering
#  num_classes:
#      number of clusters to compute (defaults to 2)
#
#  Output
#  Y:
#      Cluster assignments


    #################################################################
    # compute eigenvectors                                          #
    # U = (n x n) eigenvector matrix                                #
    # E = (n x n) eigenvalue diagonal matrix (sorted)               #
    #################################################################

    [E,U] = 

    #################################################################
    #################################################################

    #################################################################
    # compute the clustering assignment from the eigenvector        #
    # Y = (n x 1) cluster assignments [1,2,...,c]                   #
    #################################################################
    

    Y = 
    
    #################################################################
    #################################################################
    return Y









def two_blobs_clustering():
    #       a skeleton function for questions 2.1,2.2

    # load the data

    in_data =scipy.io.loadmat(path+'/data/data_2blobs')
    X = in_data['X']
    Y = in_data['Y']

    # automatically infer number of labels from samples
    num_classes = len(np.unique(Y))

    #################################################################
    # choose the experiment parameter                               #
    #################################################################

    k=
    var =  # exponential_euclidean's sigma^2

    laplacian_normalization = ''; #either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
    chosen_eig_indices =  # indices of the ordered eigenvalues to pick

    #################################################################
    #################################################################

    # build laplacian
    W=
    L =  

    Y_rec = 

    plot_clustering_result(X, Y, L, Y_rec,skc.KMeans(num_classes).fit_predict(X))
    
    
    
    
    
    
    
    
    
    
def choose_eig_function(eigenvalues):
    #  [eig_ind] = choose_eig_function(eigenvalues)
    #     chooses indices of eigenvalues to use in clustering
    #
    # Input
    # eigenvalues:
    #     eigenvalues sorted in ascending order
    #
    # Output
    # eig_ind:
    #     the indices of the eigenvectors chosen for the clustering
    #     e.g. [1,2,3,5] selects 1st, 2nd, 3rd, and 5th smallest eigenvalues

    eig_ind = 
    #################################################################
    #################################################################
    
    return eig_ind
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def spectral_clustering_adaptive(L, num_classes=2):
    #      a skeleton function to perform spectral clustering, needs to be completed
    #
    #  Input
    #  L:
    #      Graph Laplacian (standard or normalized)
    #  num_classes:
    #      number of clusters to compute (defaults to 2)
    #
    #  Output
    #  Y:
    #      Cluster assignments



    #################################################################
    # compute eigenvectors                                      #####
    # U = (n x n) eigenvector matrix                            #####
    # E = (n x n) eigenvalue diagonal matrix (sorted)           #####
    #################################################################

    [E,U] =

    #################################################################
    #################################################################


    #################################################################
    # compute the clustering assignment from the eigenvector    #####
    # Y = (n x 1) cluster assignments [1,2,...,c]               #####
    #################################################################

    Y =

    #################################################################
    #################################################################
    return Y











def find_the_bend():
    #      a skeleton function for question 2.3 and following, needs to be completed
    #

    # the number of samples to generate
    num_samples = 600

    [X, Y] = blobs(num_samples,4,0.2)

    # automatically infer number of clusters from samples
    num_classes = len(np.unique(Y));


    #################################################################
    # choose the experiment parameter                               #
    #################################################################

    k=
    var =  # exponential_euclidean's sigma^2

    laplacian_normalization = ''; #either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization

    #################################################################
    #################################################################

    # build the laplacian
    W=   
    L =  
    

    #################################################################
    # compute first 15 eigenvalues and apply                        #
    # eigenvalues: (n x 1) vector storing the first 15 eigenvalues  #
    #               of L, sorted from smallest to largest           #
    #################################################################

    [E,U] =
    eigenvalues = E[:15]
    chosen_eig_indices = choose_eig_function(eigenvalues) # indices of the ordered eigenvalues to pick

    #################################################################
    #################################################################

    #################################################################
    # compute spectral clustering solution using a non-adaptive     #
    # method first, and an adaptive one after (see handout)         #
    # Y_rec = (n x 1) cluster assignments [1,2,...,c]               #
    #################################################################

    Y_rec = 
    #Y_rec = 

    #################################################################
    #################################################################


    plot_the_bend(X, Y, L, Y_rec, eigenvalues)

    
    
    
    
    
    
    
    

   
    
    
    
    
    
def two_moons_clustering():
    #       a skeleton function for questions 2.7

    # load the data

    in_data =scipy.io.loadmat(path+'/data/data_2moons')
    X = in_data['X']
    Y = in_data['Y']

    # automatically infer number of labels from samples
    num_classes = len(np.unique(Y))

    #################################################################
    # choose the experiment parameter                               #
    #################################################################

    k=
    var = ; # exponential_euclidean's sigma^2

    laplacian_normalization = ; #either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
    chosen_eig_indices =  # indices of the ordered eigenvalues to pick

    #################################################################
    #################################################################

    # build laplacian
    W=
    L = 

    Y_rec = 

    plot_clustering_result(X, Y, L, Y_rec,skc.KMeans(num_classes).fit_predict(X))















def point_and_circle_clustering():
    #  [] = point_and_circle_clustering()
    #       a skeleton function for questions 2.8

    # load the data

    in_data =scipy.io.loadmat(path+'/data/data_pointandcircle')
    X = in_data['X']
    Y = in_data['Y']

    # automatically infer number of labels from samples
    num_classes = len(np.unique(Y))

    #################################################################
    # choose the experiment parameter                               #
    #################################################################

    k=
    var = ; # exponential_euclidean's sigma^2

    laplacian_normalization = 'rw'; #either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
    chosen_eig_indices =  # indices of the ordered eigenvalues to pick

    #################################################################
    #################################################################

    # build laplacian
    W=
    L_unn =  
    L_norm =  

    Y_unn = 
    Y_norm = 


    plot_clustering_result(X, Y, L_unn, Y_unn, Y_norm, 1);

















def parameter_sensitivity():
    # parameter_sensitivity
    #       a skeleton function to test spectral clustering
    #       sensitivity to parameter choice

    # the number of samples to generate
    num_samples = 500;

    
    #################################################################
    # choose the experiment parameter                               #
    #################################################################

    var = ; # exponential_euclidean's sigma^2

    laplacian_normalization = ; #either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
    chosen_eig_indices =  # indices of the ordered eigenvalues to pick

    parameter_candidate =   # the number of neighbours for the graph or the epsilon threshold
    parameter_performance=[]
    #################################################################
    #################################################################


    for k in parameter_candidate:

#        graph_param.graph_thresh = parameter_candidate(i); 

        [X, Y] = two_moons(num_samples,1,0.02)

        # automatically infer number of labels from samples
        num_classes = len(np.unique(Y))

        W=build_similarity_graph(X, k=k)
        L =  build_laplacian(W, laplacian_normalization='')

        Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes)

        parameter_performance+= [skm.adjusted_rand_score(Y,Y_rec)]

    plt.figure()
    plt.plot(parameter_candidate, parameter_performance)
    plt.title('parameter sensitivity')
    plt.show()
