
# coding: utf-8

# In[1]:

#package import and simplification
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split

bbs = np.loadtxt('bbs-train.txt')
label = np.loadtxt('label-train.txt')
label = label[:,1]
label_pro = np.empty([len(label),2])
for i in range(len(label)):
    if label[i] == 0:
        label_pro[i] = [1,0]
    elif label[i] == 1:
        label_pro[i] = [0,1]    
        

x_train, x_test, y_train, y_test = train_test_split(bbs, label_pro, test_size=0.2, random_state=0)
x = tf.placeholder("float", [None, 800])
y_ = tf.placeholder("float", [None, 2])


# In[2]:

#weight and bias setup
W = tf.Variable(tf.random_normal([800, 200]))
b = tf.Variable(tf.random_normal([200]))
W2 = tf.Variable(tf.random_normal([200, 2]))
b2 = tf.Variable(tf.random_normal([2]))

layer1 = tf.nn.sigmoid(tf.matmul(x,W) + b)
y= tf.nn.sigmoid(tf.matmul(layer1,W2) + b2)


#cross_entropy = -tf.reduce_sum(y_*tf.log(y+0.00001))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits= y))


#train_step = tf.train.RMSPropOptimizer(2e-5).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(2e-5).minimize(cross_entropy)
train_step = tf.train.ProximalAdagradOptimizer(2e-5).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(2e-5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

error = []
_result = []


for j in range(800):
    for i in range(12):
        random_select = np.random.randint(0,len(y_train), 2000)
        xs = [x_train[k] for k in random_select]
        ys = [y_train[k] for k in random_select] 
    
        sess.run(train_step, feed_dict={x: x_train, y_: y_train})
        train_accuracy, loss = sess.run([accuracy,cross_entropy]
                                            , feed_dict={x:x_train, y_: y_train})
    
    error.append(loss)
    
    if j%50 == 0:    
        print("step %d, training accuracy =%g"%(j, train_accuracy),"test accuracy =%g"% sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))        
        print("loss = ", loss)

train_accuracy, loss = sess.run([accuracy,cross_entropy], feed_dict={x:x_train, y_: y_train})
    


# In[ ]:




# In[ ]:



