import numpy as np
import matplotlib.pyplot as pyplot
import scipy.spatial.distance as sd
import sys
import os


sys.path.append(os.path.dirname(os.getcwd()))
from helper import *
from graph_construction.generate_data import *












def build_similarity_graph(X, var=1, eps=0, k=0):
#      Computes the similarity matrix for a given dataset of samples.
#
#  Input
#  X:
#      (n x m) matrix of m-dimensional samples
#  k and eps:
#      controls the main parameter of the graph, the number
#      of neighbours k for k-nn, and the threshold eps for epsilon graphs
#  var:
#      the sigma value for the exponential function, already squared
#
#
#  Output
#  W:
#      (n x n) dimensional matrix representing the adjacency matrix of the graph
#  similarities:
#      (n x n) dimensional matrix containing
#      all the similarities between all points (optional output)

  assert eps + k != 0, "Choose either epsilon graph or k-nn graph"


#################################################################  
#  build full graph 
#  similarities: (n x n) matrix with similarities between       
#  all possible couples of points.
#  The similarity function is d(x,y)=exp(-||x-y||^2/var)
#################################################################
  # euclidean distance squared between points
  dists = 
  similarities = 

#################################################################
#################################################################

  if eps:
#################################################################
#  compute an epsilon graph from the similarities               #
#  for each node x_i, an epsilon graph has weights              #
#  w_ij = d(x_i,x_j) when w_ij > eps, and 0 otherwise           #
#################################################################
    similarities =
     
    return similarities
    
#################################################################
#################################################################

  if k:
#################################################################
#  compute a k-nn graph from the similarities                   #
#  for each node x_i, a k-nn graph has weights                  #
#  w_ij = d(x_i,x_j) for the k closest nodes to x_i, and 0      #
#  for all the k-n remaining nodes                              #                   
#  Remember to remove self similarity and                       #
#  make the graph undirected                                    #
#################################################################
    similarities =
    
    return similarities
    
#################################################################
#################################################################













def plot_similarity_graph(X, Y):

    #      a skeleton function to analyze the construction of the graph similarity
    #      matrix, needs to be completed

    #################################################################
    #  choose the type of the graph to build and the respective     #
    #  threshold and similarity function options                    #
    #################################################################

    eps=
    var=

    #################################################################
    #################################################################

    #################################################################
    # use the build_similarity_graph function to build the graph W  #
    # W: (n x n) dimensional matrix representing                    #
    #    the adjacency matrix of the graph                          #
    #################################################################

    W=

    #################################################################
    #################################################################

    plot_graph_matrix(X,Y,W)
    
    #################################################################
    #################################################################
    
    
    
    
    
    










def how_to_choose_epsilon():
#       a skeleton function to analyze the influence of the graph structure
#       on the epsilon graph matrix, needs to be completed


# the number of samples to generate
    num_samples = 100
    
    #################################################################
    # the option necessary for worst_case_blob, try different       #
    # values                                                        #
    #################################################################

    gen_pam =   # read worst_case_blob.m to understand the meaning of
    #                               the parameter

    #################################################################
    #################################################################
    [X, Y] = worst_case_blob(num_samples,gen_pam)
    
    #################################################################
    # use the similarity function and the max_span_tree function    #
    # to build the maximum spanning tree max_tree                   #
    # sigma2: the exponential_euclidean's sigma2 parameter          #
    # similarities: (n x n) matrix with similarities between        #
    #              all possible couples of points                   #
    # max_tree: (n x n) indicator matrix for the edges in           #
    #           the maximum spanning tree                           #
    #################################################################

    var =   # exponential_euclidean's sigma^2

    dists = 
    similarities = 
    
    max_tree = 
    #################################################################
    #################################################################

    #################################################################
    # set graph_thresh to the minimum weight in max_tree            #
    #################################################################

    eps = 
    
    
    #################################################################
    #################################################################


    #################################################################
    # use the build_similarity_graph function to build the graph W  #
    # W: (n x n) dimensional matrix representing                    #
    #    the adjacency matrix of the graph  
    #    use plot_graph_matrix to plot the graph                    #
    #################################################################

 
    #################################################################
    #################################################################
    
    
