import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

#处理notmnist数据

def get_file(file_dir):

    images = []
    for root, sub_folders, files in os.walk(file_dir):
        # image directories
        for name in files:
            images.append(os.path.join(root, name))
    # assign 10 labels based on the folder names
    labels = []
    for one_folder in images:

        letter = one_folder.split('\\')[-2]
        te=np.zeros(10)
        # 通过子文件夹名字获取标签，标签编号为1-10

        if letter == 'A':
            te[0]=1
            labels.append(te)
        elif letter == 'B':
            te[1] = 1
            labels.append(te)
        elif letter == 'C':
            te[2] = 1
            labels.append(te)
        elif letter == 'D':
            te[3] = 1
            labels.append(te)
        elif letter == 'E':
            te[4] = 1
            labels.append(te)
        elif letter == 'F':
            te[5] = 1
            labels.append(te)
        elif letter == 'G':
            te[6] = 1
            labels.append(te)
        elif letter == 'H':
            te[7] = 1
            labels.append(te)
        elif letter == 'I':
            te[8] = 1
            labels.append(te)
        else:
            te[9] = 1
            labels.append(te)

    #下载数据转换为数组
    label_list=np.array(labels)
    for i in range(len(images)):
        try:

            img = Image.open(images[i])
            images[i]=np.array(img).reshape(784)
            # print(i, 'succ',images[i] )
        except:
            print(i, "can't open")

#混淆
    image_list=np.array(images)
    np.random.seed(110)
    np.random.shuffle(image_list)
    np.random.seed(110)
    np.random.shuffle(label_list)
#测试
    # fig, ax = plt.subplots(4, 4, figsize=(15, 15))
    # fig.subplots_adjust(wspace=0.1, hspace=0.7)
    # for i in range(4):
    #     for j in range(4):
    #         ax[i, j].imshow(image_list[i * 10 + j].reshape(28, 28))
    #         # 用argmax函数取出z3中最大的数的序号，即为预测结果：
    #         predicted_num = label_list[i * 10 + j]
    #         # 这里不能用tf.argmax，因为所有的tf操作都是在图中，没法直接取出来
    #         ax[i, j].set_title('Predict:' + str(predicted_num))
    #         ax[i, j].axis('off')
    # plt.show()

    return image_list, label_list

get_file(r'C:\Users\ming\PycharmProjects\cnn_test\notMNIST_small')