
# coding: utf-8

# In[37]:

import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.cluster import KMeans
os.chdir('C:/Users/12562/Desktop/final')
IMG1=np.loadtxt('imgs_sample_1.txt')
IMG2=np.loadtxt('imgs_sample_2.txt')
IMG3=np.loadtxt('imgs_sample_3.txt')
IMG4=np.loadtxt('imgs_sample_4.txt')
IMG5=np.loadtxt('imgs_sample_5.txt')
IMG6=np.loadtxt('imgs_sample_6.txt')
IMG7=np.loadtxt('imgs_sample_7.txt')
ECFP=np.loadtxt('ECFPs.txt') 
#a=np.column_stack((IMG1,IMG2)) concatenate through 
vector=np.concatenate([IMG1,IMG2,IMG3,IMG4,IMG5,IMG6,IMG7])


# In[40]:

kmeans = KMeans(n_clusters=32, random_state=0)
clusters = kmeans.fit_predict(vector)
kmeans.cluster_centers_.shape


# In[52]:

fig, ax = plt.subplots(2, 14, figsize=(20, 10))
centers = kmeans.cluster_centers_.reshape(32, 60, 12)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



