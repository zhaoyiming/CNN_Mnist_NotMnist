from tensorflow.keras.models import Sequential
import tensorflow.keras
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier




import tensorflow.keras.backend as K

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten
import inputminst
def data():
    X, Y = inputminst.get_file(r'C:\Users\ming\PycharmProjects\cnn_test\notMNIST_small')
    X_train = X[:1000]
    X_train = X_train.reshape(-1, 28, 28, 1)
    Y_train = Y[:1000]

    X_test = X[-50:-1]
    X_test = X_test.reshape(-1, 28, 28, 1)
    Y_test = Y[-50:-1]
    return X_train, Y_train, X_test, Y_test


def generator(x_train,y_train,batch_size):
    while 1:
        row = np.random.randint(0,len(x_train),size=batch_size)
        x = np.zeros((batch_size,x_train.shape[-1]))
        y = np.zeros((batch_size,))
        x = x_train[row]
        y = y_train[row]
        yield x,y


def cnn(channel,height,width,classes):
    inputshape=(channel,height,width)

    if K.image_data_format() == "channels_last":  # 确认输入维度,就是channel是在开头，还是结尾
        inputshape = (height, width, channel)
    model=Sequential()
    model.add(Conv2D(6,(5,5),padding="same",activation="relu",input_shape=inputshape,name="conv1"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="pool1"))

    model.add(Conv2D(16, (5, 5), padding="same", activation="relu", input_shape=inputshape, name="conv2"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2"))

    model.add(Flatten())
    model.add(Dense(512,activation="relu",name="fc1"))

    model.add(Dropout(0.9))
    model.add(Dense(classes,activation="softmax",name="fc2"))

    return model
def train(model,train_x,train_y,test_x,test_y,epochs,batchsize):
    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",metrics=["accuracy"])
    #输出训练指标
    _history=model.fit_generator(generator(train_x,train_y,batchsize),
                        validation_data=(test_x,test_y),steps_per_epoch=len(train_x)/batchsize,
                        epochs=epochs,verbose=1)

    score = model.evaluate(test_x, test_y)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])



    #绘制图像
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0,N),_history.history["loss"],label ="train_loss")
    plt.plot(np.arange(0,N),_history.history["val_loss"],label="val_loss")
    plt.plot(np.arange(0,N),_history.history["acc"],label="train_acc")
    plt.plot(np.arange(0,N),_history.history["val_acc"],label="val_acc")
    plt.title("loss and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("loss/acc")
    plt.legend(loc="best")
    plt.savefig("./result.png")
    plt.show()




if __name__ =="__main__":
    model=cnn(channel=1, height=28,
                         width=28, classes=10)
    # #数据增强_迭代器
    # aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    #                          height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    #                          horizontal_flip=True, fill_mode="nearest")  # 数据增强，生成迭代器

    #普通迭代器
    batchsize=64
    epochnum=40
    X_train, Y_train, X_test, Y_test=data()
    train(model,X_train,Y_train,X_test,Y_test,epochnum,batchsize)




