#import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from datetime import timedelta
import math
#根据LeNet5网络设计的针对手写数字Mnist和手写字母NotMnist的CNN


#input
import inputminst
X,Y=inputminst.get_file(r'C:\Users\ming\PycharmProjects\cnn_test\notMNIST_small')
X_train=X[:15000]
Y_train=Y[:15000]
X_test=X[-3000:-1]
Y_test=Y[-3000:-1]


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.1,shape=length))
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(inputx):
    return tf.nn.max_pool(inputx,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#mnist数据集输入
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('mnist/', one_hot=True)
# X=mnist.train.images
# Y=mnist.train.labels








datacls = np.argmax(Y_test,axis=1)   # show the real test labels:  [7 2 1 ..., 4 5 6], 10000values
#占位符
x = tf.placeholder("float",shape=[None,784],name='x')
x_image = tf.reshape(x,[-1,28,28,1])

y_true = tf.placeholder("float",shape=[None,10],name='y_true')
y_true_cls = tf.argmax(y_true,dimension=1)

#开始设计卷积层
# Conv 1
layer_conv1 = {"weights":new_weights([5,5,1,6]),
               "biases":new_biases([6])}
h_conv1 = tf.nn.relu(conv2d(x_image,layer_conv1["weights"])+layer_conv1["biases"])
h_pool1 = max_pool_2x2(h_conv1)
# Conv 2
layer_conv2 = {"weights":new_weights([5,5,6,16]),
               "biases":new_biases([16])}
h_conv2 = tf.nn.relu(conv2d(h_pool1,layer_conv2["weights"])+layer_conv2["biases"])
h_pool2 = max_pool_2x2(h_conv2)
# Full-connected layer 1
fc1_layer = {"weights":new_weights([7*7*16,120]),
            "biases":new_biases([120])}
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,fc1_layer["weights"])+fc1_layer["biases"])

# Full-connected layer 2
fc2_layer = {"weights":new_weights([120,10]),
             "biases":new_weights([10])}
# Predicted class
y_pred = tf.nn.softmax(tf.matmul(h_fc1,fc2_layer["weights"])+fc2_layer["biases"])  # The output is like [0 0 1 0 0 0 0 0 0 0]
y_pred_cls = tf.argmax(y_pred,dimension=1)  # Show the real predict number like '2'
# cost function to be optimized
cross_entropy = -tf.reduce_mean(y_true*tf.log(y_pred))

#调节学习率
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)
# Performance Measures
correct_prediction = tf.equal(y_pred_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

#批处理
def next_batch(train_data, train_target, batch_size):
    index = [ i for i in range(0,len(train_target)) ]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0,batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target
def optimize(num_iterations):
    start_time = time.time()

    #开始训练
    for i in range(0,num_iterations):
        x_batch,y_batch = next_batch(X_train,Y_train,50)
        sess.run(optimizer,feed_dict={x:x_batch,y_true:y_batch})
        # Print status every 100 iterations.
        if i%100==0:
            acc = sess.run(accuracy,feed_dict={x:x_batch,y_true:y_batch})

            msg = "Optimization Iteration:{0:>6}, Training Accuracy: {1:>6.1%}"
            _,batch_cost = sess.run([optimizer,cross_entropy], feed_dict={x:x_batch,y_true:y_batch})
            print(batch_cost)
            print(msg.format(i+1,acc))


    end_time = time.time()
    time_dif = end_time-start_time
    print("Time usage:"+str(timedelta(seconds=int(round(time_dif)))))

#测试正确率
test_batch_size = 256
def print_test_accuracy():
    # Number of images in the test-set.
    num_test = len(X_test)
    cls_pred = np.zeros(shape=num_test,dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i+test_batch_size,num_test)
        images = X_test[i:j, :]
        labels = Y_test[i:j, :]
        feed_dict={x:images,y_true:labels}
        cls_pred[i:j] = sess.run(y_pred_cls,feed_dict=feed_dict)
        i = j
    cls_true = datacls
    correct = (cls_true==cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    # Print the accuracy
    msg = "Accuracy on Test-Set: {0:.1%} ({1}/{2})"
    print(msg.format(acc,correct_sum,num_test))
# Performance after 10000 optimization iterations
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    optimize(10000)
    print_test_accuracy()