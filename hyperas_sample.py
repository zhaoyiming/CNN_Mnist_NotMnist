from tensorflow.keras.models import Sequential
import tensorflow.keras
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten
import inputminst
def data():
    X, Y = inputminst.get_file(r'C:\Users\ming\PycharmProjects\cnn_test\notMNIST_small')
    x_train = X[:1000]
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = Y[:1000]

    x_test = X[-50:-1]
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = Y[-50:-1]
    return x_train, y_train, x_test, y_test




def create_model(x_train,y_train,x_test,y_test):
    inputshape = (1, 28, 28)
    epochs=40
    batchsize=64

    if K.image_data_format() == "channels_last":  # 确认输入维度,就是channel是在开头，还是结尾
        inputshape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding="same", activation="relu", input_shape=inputshape, name="conv1"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1"))

    model.add(Conv2D(16, (5, 5), padding="same", activation="relu", input_shape=inputshape, name="conv2"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2"))

    model.add(Flatten())
    model.add(Dense({{choice([256, 512, 1024])}}, activation="relu", name="fc1"))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(10, activation="softmax", name="fc2"))


    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",metrics=["accuracy"])
    #输出训练指标
    result=model.fit(x_train,y_train, batch_size={{choice([64, 128])}},
                        validation_data=(x_test,y_test),steps_per_epoch=len(x_train)/batchsize,
                        epochs=epochs,verbose=0)

    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}





if __name__ =="__main__":



    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    train_x, train_y, test_x, test_y = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(test_x, test_y))
    print(best_run)



