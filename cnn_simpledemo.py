import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import inputminst


# mnist导入
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('mnist/', one_hot=True)
# X_train,Y_train = mnist.train.images,mnist.train.labels
# X_test,Y_test = mnist.test.images,mnist.test.labels

#Notmnist导入
X,Y=inputminst.get_file(r'C:\Users\ming\PycharmProjects\cnn_test\notMNIST_small')
X_train=X[:15000]
Y_train=Y[:15000]
X_test=X[-3000:-1]
Y_test=Y[-3000:-1]



# 数据的形状：
print(X_train.shape)  # (55000, 784)
print(Y_train.shape)  # (55000, 10)
print(X_test.shape)   # (10000, 784)
print(Y_test.shape)   # (10000, 10)

#无实际意义
tf.reset_default_graph()

#占位符
X=tf.placeholder(dtype=tf.float32,shape=[None,784],name='X')
Y=tf.placeholder(dtype=tf.float32,shape=[None,10],name='Y')

#创建神经网络各层
W1=tf.get_variable('W1',[784,128],initializer=tf.contrib.layers.xavier_initializer())
b1=tf.get_variable('b1',[128],initializer=tf.zeros_initializer())
W2=tf.get_variable('W2',[128,64],initializer=tf.contrib.layers.xavier_initializer())
b2=tf.get_variable('b2',[64],initializer=tf.zeros_initializer())
W3=tf.get_variable('W3',[64,10],initializer=tf.contrib.layers.xavier_initializer())
b3=tf.get_variable('b3',[10],initializer=tf.zeros_initializer())

#计算loss
A1=tf.nn.relu(tf.matmul(X,W1)+b1,name='A1')
A2=tf.nn.relu(tf.matmul(A1,W2)+b2,name='A2')
Z3=tf.matmul(A2,W3)+b3
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
#最小化loss
trainer=tf.train.AdamOptimizer().minimize(cost)

#批量处理
def next_batch(train_data, train_target, batch_size):
    index = [ i for i in range(0,len(train_target)) ]
    np.random.shuffle(index);
    batch_data = [];
    batch_target = [];
    for i in range(0,batch_size):
        batch_data.append(train_data[index[i]]);
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    costs=[]
    #训练循环
    for i in range(15000):
        X_batch,Y_batch=next_batch(X_train,Y_train,64)
        _,batch_cost=sess.run([trainer,cost],feed_dict={X:X_batch,Y:Y_batch})
        costs.append(batch_cost)
        #每100次打印loss
        if i%100==0:
            print('iter%d,batch_cost'%i,batch_cost)

    predictions=tf.equal(tf.argmax(tf.transpose(Z3)),tf.argmax(tf.transpose(Y)))
    accuracy=tf.reduce_mean(tf.cast(predictions,'float'))

    print("train accuracy:",sess.run(accuracy,feed_dict={X:X_train,Y:Y_train}))
    print("test accuracy:", sess.run(accuracy, feed_dict={X: X_test, Y: Y_test}))

    #检验可视化
    z3, acc = sess.run([Z3, accuracy], feed_dict={X: X_test, Y: Y_test})
    print("Test set accuracy:", acc)

    # 随机从测试集中抽一些图片（比如第i*10+j张图片），然后取出对应的预测（即z3[i*10+j]）：
    fig, ax = plt.subplots(4, 4, figsize=(15, 15))
    fig.subplots_adjust(wspace=0.1, hspace=0.7)
    for i in range(4):
        for j in range(4):
            ax[i, j].imshow(X_test[i * 10 + j].reshape(28, 28))
            # 用argmax函数取出z3中最大的数的序号，即为预测结果：
            predicted_num = np.argmax(z3[i * 10 + j])
            # 这里不能用tf.argmax，因为所有的tf操作都是在图中，没法直接取出来
            ax[i, j].set_title('Predict:' + str(predicted_num))
            ax[i, j].axis('off')
    plt.show()


