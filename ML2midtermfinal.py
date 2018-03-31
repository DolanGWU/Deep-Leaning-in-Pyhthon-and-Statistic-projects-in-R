
# coding: utf-8

# In[2]:

#pre-code
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import os;
print (os.getcwd()) 


# In[8]:

#Batch Making for data file 
#create a while loop and read the text file line by line
import numpy as np
import pickle
with open('bbs-train.txt') as f:
    count = 0
    batch = []
    counter = 0
    for line in f:
        if count<150:
            batch.append(np.array(line.strip().split(" "),dtype=np.float64))
            count = count + 1
        else:
            filen = "C:\\Users\\12562\\batches\\datafile"+str(counter)
            pickle.dump(batch,open(filen,"wb"))
            #with open('batches\\datafile' + str(counter),"w") as h:
            #    h.write("\n".join(str(batch)))
            #print(str(batch[0]))
            counter = counter + 1
            #write batch into a file
            #label_{counter}
            batch.clear() 
            count = 0
#print(batch)


# In[9]:

#Batch making for labels 
#create a while loop and read the text file line by line
import numpy as np
with open('label-train.txt') as d:
    count = 0
    batch = []
    counter = 0
    for line in d:
        if count<150:
            batch.append(int(line.strip().split("   ")[1]))
            count = count + 1
        else:
            filen = "C:\\Users\\12562\\batches\\belfile"+str(counter)
            pickle.dump(batch,open(filen,"wb"))
            counter = counter + 1
            #write batch into a file
            #label_{counter}
            batch.clear() 
            count = 0
#print(batch)


# In[10]:

#Batch making test data and label  
import pickle
import numpy as np
listdata=[]
listlabel=[]
for counter in range(24,30):
    batch_xtest = np.array(pickle.load(open("C:\\Users\\12562\\batches\\datafile"+str(counter),"rb")))
    batch_ytest = np.array(pickle.load(open("C:\\Users\\12562\\batches\\belfile"+str(counter),"rb"))).reshape(-1,1)
    listdata.append(batch_xtest)
    listlabel.append(batch_ytest)
X_test = np.vstack(listdata)
filen = "C:\\Users\\12562\\batches\\testdatafile"
pickle.dump(X_test,open(filen,"wb"))
Y_test = np.vstack(listlabel)
filen = "C:\\Users\\12562\\batches\\testlabelfile"
pickle.dump(Y_test,open(filen,"wb"))


# In[11]:

#Model making with batched data
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
import numpy as np
import pickle
x = tf.placeholder("float", [None, 800])
y_ = tf.placeholder("float", [None,1])

# inference
W = tf.Variable(tf.random_normal([800, 300]))
b = tf.Variable(tf.random_normal([ 300]))

W2 = tf.Variable(tf.random_normal([300, 1]))
b2 = tf.Variable(tf.random_normal([1]))

#matm=tf.matmul(x,W)
layer1 = tf.nn.sigmoid(tf.matmul(x,W) + b)
y= tf.nn.sigmoid(tf.matmul(layer1,W2) + b2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y+0.00001))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.round(y), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
   # print("epoch"+str(i))
    #read the batch one by one from the folder
    #counter=0
    for counter in range(23):
        #print(counter)
        batch_x = np.array(pickle.load(open("C:\\Users\\12562\\batches\\datafile"+str(counter),"rb")))
        batch_y = np.array(pickle.load(open("C:\\Users\\12562\\batches\\belfile"+str(counter),"rb"))).reshape(-1,1)
        #print(batch_x.shape)
        #print(batch_y.shape)
        sess.run([train_step,cross_entropy], feed_dict={x: batch_x, y_: batch_y})
    if i%2 == 0:
        batch_x = np.array(pickle.load(open("C:\\Users\\12562\\batches\\testdatafile","rb")))
        batch_y = np.array(pickle.load(open("C:\\Users\\12562\\batches\\testlabelfile","rb"))).reshape(-1,1)
        loss,acc = sess.run([cross_entropy,accuracy], feed_dict={x: batch_x, y_: batch_y})
        print(str(i)+":")
        print(loss)
        print(acc)
        print()
    #print(sess.run([cross_entropy]))
        #counter = counter + 1
print("done")


# In[ ]:



