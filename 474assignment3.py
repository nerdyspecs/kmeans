#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import math
import random

def kmeans(k,epsilon):

    df = pd.read_csv('winequality-white.csv')

    # calculatin euclidean distance 
    def euclidean(a,b):
        target = a.tolist()
        other = b.tolist()
        if len(a) != len(b):
            if len(a) > len(b):
                temp = target
                target = other
                other = temp
            for x in range(0, abs(len(a) - len(b))):
                target.append(0)
        assert(len(target) == len(other))
        total = 0
        for x in range(0, len(target)):
            total += (target[x] - other[x])**2
        return math.sqrt(total)

    # generating number of K's and number of clusters
    def generateK(a):
        ks = []
        clusters = []
        for x in range(a):
            temp = random.randint(1,df.shape[0])
            if (temp in ks == True):
                printf("Its da same")
                while(temp in ks):
                    temp = random.randint(a,df.shape[0])
            ks.append(temp)
        return ks

    # Converting df into numpy arrays.
    X = np.array(df)

    #Generating initial random centroids
    centroids = []
    index = generateK(k);

    #appending first centroids to centroid list
    for x in index: 
        centroids.append(X[x]) 

    status = False
    iteration = 0


    #starts here
    while (status == False ):
        iteration += 1
        newCentroids = []

        # creatin clusters.
        clusters = []    
        for x in range(len(centroids)):
            cluster = []
            clusters.append(cluster)

        for x in X:
            temp = []
            for centroid in centroids:
                temp.append(euclidean(x,centroid))
            minposition = temp.index(min(temp))
            clusters[minposition].append(x)

        # creating newCentroids
        for x in range(len(centroids)):
            newCentroids.append(np.array(clusters[x]).mean(axis=0))
        count = 0
        for x in range(len(centroids)):
            if (euclidean(np.array(newCentroids[x]),np.array(centroids[x])) < epsilon):
                count += 1

        if (count == len(centroids)):
            status = True
        else:
            centroids = newCentroids
            
    print("Iterations :",iteration)      
    for x in range(len(clusters)):
        print("Cluster [",x,"] =",len(clusters[x])," ",str(round(((len(clusters[x]))/X.shape[0])*100,2))+"%")

