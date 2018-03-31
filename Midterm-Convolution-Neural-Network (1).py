
# coding: utf-8

# In[9]:

import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

bbs = np.loadtxt('bbs-train.txt') #opens file with name of "test.txt"
label = np.loadtxt('label-train.txt')
label = label[:,1]
label_dummy = np.empty([len(label),2])

for i in range(len(label)):
    if label[i] == 0:
        label_dummy[i] = [1,0]
    elif label[i] == 1:
        label_dummy[i] = [0,1]    
#create a while loop and read the tex


# In[181]:

#creating batches
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

bbs = open("bbs-train.txt","r") #opens file with name of "test.txt"
label = open("label-train.txt","r")
#create a while loop and read the text file line by line
import numpy as np
import pickle
with open('bbs-train.txt') as f:
    count = 0
    batch = []
    counter = 0
    for line in f:
        if count<100:
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


# In[182]:

#creating batches for label
#create a while loop and read the text file line by line
import numpy as np
with open('label-train.txt') as d:
    count = 0
    batch = []
    counter = 0
    for line in d:
        if count<100:
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


# In[183]:

#making test file 
import pickle
import numpy as np
listdata=[]
listlabel=[]
for counter in range(33,45):
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

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
import numpy as np
import pickle
x = tf.placeholder(np.float32, [None, 800],name="x")
y_ = tf.placeholder(np.float32, [None,1],name="y")
x_image = tf.reshape(x, [-1,40,20,1])
x_image = tf.image.resize_bilinear(x_image, size=[40,40])


def weight_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# inference
#W = tf.Variable(tf.random_normal([800, 300]))
#b = tf.Variable(tf.random_normal([ 300]))

W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])



W_fc1 = weight_variable([10 * 10 * 64, 500])
b_fc1 = bias_variable([500])

W_fc2 = weight_variable([500, 1])
b_fc2 = bias_variable([1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * 64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")


h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#W2 = tf.Variable(tf.random_normal([300, 100]))
#b2 = tf.Variable(tf.random_normal([100]))

#matm=tf.matmul(x,W)
#layer1 = tf.nn.sigmoid(tf.matmul(x,W) + b)
#y= tf.nn.sigmoid(tf.matmul(layer1,W2) + b2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y+0.001))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.round(y-0.01), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

prob=0.5
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
   # print("epoch"+str(i))
    #read the batch one by one from the folder
    #counter=0
    for counter in range(32):
        #print(counter)
        batch_x = np.array(pickle.load(open("C:\\Users\\12562\\batches\\datafile"+str(counter),"rb")))
        batch_y = np.array(pickle.load(open("C:\\Users\\12562\\batches\\belfile"+str(counter),"rb")),dtype="float").reshape(-1,1)
        #batch_y = np.vstack(batch_y)
        #print(type(batch_x[0][0]))
        #print(type(batch_y[0][0]))
        #print(sess.run([y_,train_step], feed_dict={x: batch_x, y_: batch_y}))
        sess.run([train_step], feed_dict={x: batch_x, y_: batch_y,keep_prob:prob})

    if i%2 == 0:
        batch_x=np.array(pickle.load(open("C:\\Users\\12562\\batches\\testdatafile","rb")))
        batch_y= np.array(pickle.load(open("C:\\Users\\12562\\batches\\testlabelfile","rb")),dtype="float").reshape(-1,1)
        loss,acc = sess.run([cross_entropy,accuracy], feed_dict={x: batch_x, y_: batch_y,keep_prob:1})
        print(str(i)+":")
        print(loss)
        print(acc)
        print()
    #print(sess.run([cross_entropy]))
        #counter = counter + 1

print("done")


# In[10]:

filen = "C:\\Users\\12562\\batches\\confcbias1file"+str(counter)
pickle.dump(sess.run(b_fc1),open(filen,"wb"))


# In[10]:

print(label)


# In[11]:


print(bbs)


# In[ ]:

import tensorflow as tf
from utils import createCheckPoints,optimize,print_test_accuracy,initializeWeights_Bias,initialize_x_y,initializeParameters,createDataDict,convertToDataset,plot_confusion_matrix
from dataLoad import dataLoad

def fitSimpleNN(img_size_flat,num_classes,data,parameters):
    x, y_true, y_true_cls = initialize_x_y(img_size_flat, num_classes)
    weights, biases = initializeWeights_Bias(img_size_flat, num_classes)

    logits = tf.matmul(x, weights) + biases
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=y_true)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdagradOptimizer(parameters['training_epochs']).minimize(loss)


    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    tfObject = {'x': x, 'y_true': y_true, 'optimizer': optimizer, 'y_pred_cls': y_pred_cls, 'accuracy': accuracy,
                'loss': loss, 'saver': saver}
    train_data = convertToDataset(data,parameters['batch_size'])

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path, run_optimize = createCheckPoints(session, saver, 'Logistic Reg')
    parameters['save_path'] = save_path
    if run_optimize:
        optimize(parameters,train_data,data,tfObjects=tfObject,session = session)

    pred_labels,acc = print_test_accuracy(data, 250, tfObject, session)
    session.close()
    return pred_labels,acc

if __name__=="__main__":
    bbs_train, imgs_train, labels = dataLoad()
    imgs_size_flat = 60 * 12
    bbs_size_flat = 40 * 20
    num_classes = 2
    # bbs_data = createDataDict(bbs_train, labels)
    # parameters = initializeParameters(learning_rate=0.0001, training_epochs=500, batch_size=500,display_size=50)
    # fitSimpleNN(bbs_size_flat, num_classes, bbs_data,parameters)
    imgs_data = createDataDict(imgs_train,labels)
    parameters = initializeParameters(learning_rate=0.0001, training_epoch


# In[ ]:



