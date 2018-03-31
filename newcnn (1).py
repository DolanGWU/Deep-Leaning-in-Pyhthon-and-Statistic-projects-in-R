
# coding: utf-8

# In[23]:

#data cleaning and reshaping process
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import os;
print (os.getcwd()) 
bbs = np.loadtxt('bbs-train.txt') #opens file with name of "test.txt"
label = np.loadtxt('label-train.txt')
label = label[:,1]
label_pro = np.empty([len(label),2])
for i in range(len(label)):
    if label[i] == 0:
        label_pro[i] = [1,0]
    elif label[i] == 1:
        label_pro[i] = [0,1]    
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bbs, label_pro, test_size=0.2, random_state=0)
x = tf.placeholder("float", [None, 800])
y_ = tf.placeholder("float", [None, 2])
xshape = tf.reshape(x, [-1,40,20,1])        


# In[24]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[25]:

W_conv1 = weight_variable([10, 10, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(xshape, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([10, 10, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([10 * 5 * 64, 250])
b_fc1 = bias_variable([250])

h_pool2_flat = tf.reshape(h_pool2, [-1, 10*5*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([250, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv2 = tf.argmax(y_conv,1)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits= y_conv ))

train_step = tf.train.AdamOptimizer(2e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[21]:

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for img in x_train:
    img *= 255.0/img.max()
    
for img in x_test:
    img *= 255.0/img.max()

#batch_size = 4000
error = []
_result = []

for j in range(600):
    for i in range(10):
        random_select = np.random.randint(0,len(y_train), 2000)
        xs = [x_train[k] for k in random_select]
        ys = [y_train[k] for k in random_select] 
    
        sess.run(train_step, feed_dict={x: x_train, y_: y_train,keep_prob: 0.8})
        train_accuracy, loss, y_soft = sess.run([accuracy,cross_entropy, y_conv2]
                                            , feed_dict={x:x_train, y_: y_train, keep_prob: 1})
    
    _result.append(y_soft)
    error.append(loss)
    
    if j%50 == 0:    
        print("step %d, training accuracy %g"%(j, train_accuracy))
        print("test accuracy %g"% sess.run(accuracy, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))
        print("loss= ", loss)
        
train_accuracy, loss, y_soft, tf = sess.run([accuracy,cross_entropy, y_conv2, correct_prediction]
                                            , feed_dict={x: x_test, y_: y_test, keep_prob: 1})


# In[ ]:



