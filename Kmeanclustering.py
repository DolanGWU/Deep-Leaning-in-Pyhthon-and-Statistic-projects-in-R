
# coding: utf-8

# In[20]:

# this drastic code change is after consulting with another team from from ML2 class after professor Chen's request.

import os
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from random import choice,shuffle
from numpy import array
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def KMeansCluster(vectors, clusters):

    clusters = int(clusters)
    assert clusters < len(vectors)
    dim = len(vectors[0])
    vector_indices = shuffle(list(range(len(vectors))))
    graph = tf.Graph()
                                  
    with graph.as_default():
        
        cent_assigns = []
        centroids = [tf.Variable((vectors[vector_indices[i]]))
            for i in range(clusters)]
        centval = tf.placeholder("float64", [dim])       
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centval))
        
        assignments = [tf.Variable(0) for i in range(len(vectors))]
        assignment_value = tf.placeholder("int32")

        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))
        mean_input = tf.placeholder("float", [None, dim])
        mean_op = tf.reduce_mean(mean_input, 0)
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        cetdist = tf.placeholder("float", [clusters])
        
        cluster_assignment = tf.argmin(cetdist, 0)
        euclidian = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(
            v1, v2), 2)))
        
       
        init_op = tf.global_variables_initializer()
        
        sess = tf.Session()
        sess.run(init_op)
        
        noofiterations = 10
        for iteration_n in range(noofiterations):
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                distances = [sess.run(euclidian, feed_dict={
                    v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]
                assignment = sess.run(cluster_assignment, feed_dict = {
                    cetdist: distances})
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})
            for cluster_n in range(clusters):
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                new_location = sess.run(mean_op, feed_dict={
                    mean_input: array(assigned_vects)})
                sess.run(cent_assigns[cluster_n], feed_dict={
                    centval: new_location})

        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments,distances


# In[21]:

srcdata=np.array(np.loadtxt('bbs-train.txt'))
k=200
#k=500
#k=100
center,result,distances=KMeansCluster(srcdata,k)
print (center)

res={"k":[],"count":[],"distance":[]}    
for i in range(k):
    res["k"].append(i)
    res["count"].append(result.count(i))
res["distance"]=distances

print (res)  
pdres = pd.DataFrame(res)
sns.lmplot("k","count",data=pd_res)
plt.show()


# In[ ]:




# In[ ]:



