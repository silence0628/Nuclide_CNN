# -*- coding: utf-8 -*-
"""
    include： load_data
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import argparse
import scipy.io as scio

def get_train_files(file_dir):
    Co60 = []
    label_Co60 = []
    Cs137 = []
    label_Cs137 = []
    Eu152 = []
    label_Eu152 = []
    CoCs = []
    label_CoCs = []
    CoEu = []
    label_CoEu = []
    CsEu = []
    label_CsEu = []

    # 载入数据路径并写入标签值
    for file in os.listdir(file_dir):
        name = file.split(sep='_')
        if name[0] == 'Co60':
            Co60.append(file_dir + file)
            label_Co60.append(0)
        elif name[0] == 'Cs137':
            Cs137.append(file_dir + file)
            label_Cs137.append(1)
        elif name[0] == 'Eu152':
            Eu152.append(file_dir + file)
            label_Eu152.append(2)
        elif name[0] == 'CoCs':
            CoCs.append(file_dir + file)
            label_CoCs.append(3)
        elif name[0] == 'CoEu':
            CoEu.append(file_dir + file)
            label_CoEu.append(4)
        elif name[0] == 'CsEu':
            CsEu.append(file_dir + file)
            label_CsEu.append(5)

    print('There are %d Cs137\n'
          'There are %d Co60\n'
          'There are %d Eu152\n'
          'There are %d CoCs\n'
          'There are %d CoEu\n'
          'There are %d CsEu'%(len(Cs137), len(Co60), len(Eu152), len(CoCs), len(CoEu), len(CsEu)))

    # 打乱文件顺序
    image_list = np.hstack((Co60, Cs137, Eu152, CoCs, CoEu, CsEu))
    label_list = np.hstack((label_Co60, label_Cs137, label_Eu152, label_CoCs, label_CoEu, label_CsEu))
    # print(label_list)
    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]  # ValueError: invalid literal for int() with base 10: '0.0' ==>>>https://blog.csdn.net/lanchunhui/article/details/51029124

    return image_list, label_list

def get_train_batch(image, label, image_W, image_H, BATCH_SIZE, CAPACITY):
    '''
    Args:
        image,label: 要生成batch的图像和标签list
        image_W，image_H:图片的宽、高
        batch_size: 每个batch有多少张图片
        capacity: 队列容量
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 1], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    """将python.list类型转换为tf能够识别的格式"""
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    # get  [<tf.Tensor 'input_producer/Gather:0' shape=() dtype=string>,<tf.Tensor 'input_producer/Gather_1:0' shape=() dtype=int32>]

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=1)
    ######################################
    # data argumentation should go to here
    ######################################
    #    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.resize_images(image, [image_H, image_W],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # tf.image.ResizeMethod.NEAREST_NEIGHBOR
    image = tf.cast(image, tf.float32)

    image = tf.image.per_image_standardization(image)  # 标准化图像

    #    image_batch, label_batch = tf.train.batch([image, label],
    #                                                batch_size= batch_size,
    #                                                num_threads= 64,
    #                                                capacity = capacity)
    # you can also use shuffle_batch
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=BATCH_SIZE,
                                                      num_threads=64,
                                                      capacity=CAPACITY,
                                                      min_after_dequeue=CAPACITY - 1)
    label_batch = tf.reshape(label_batch, [BATCH_SIZE])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

def get_test_files(file_path):
    animals = []
    # 载入数据路径
    for file in os.listdir(file_path):
        animals.append(file_path + file)
    return animals

def get_test_batch(image, image_W, image_H, BATCH_SIZE, CAPACITY):

    image = tf.cast(image, tf.string)
    input_queue = tf.train.slice_input_producer([image])
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [image_H, image_W],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # tf.image.ResizeMethod.NEAREST_NEIGHBOR
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)  # 标准化图像


    image_batch  = tf.train.shuffle_batch([image],
                                          batch_size=BATCH_SIZE,
                                          num_threads=64,
                                          capacity=CAPACITY,
                                          min_after_dequeue=CAPACITY - 1)

    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch



if __name__ == '__main__':

    CAPACITY = 256
    IMG_W = 32
    IMG_H = 32

    train_dir = '/home/sjj/Spectra_train/'
    batch_size = 20
    visualize = True

    print('{:*^70}'.format('【Start testing!】'))
    train, train_label = get_train_files(train_dir)
    print(len(train))
    train_batch, label_batch = get_train_batch(train, train_label, IMG_W, IMG_H, batch_size, CAPACITY)
    print(train_batch.shape)
    train_new_batch = tf.reshape(train_batch, [20, 1, 1024, 1])
    print(train_new_batch.shape)

    """开启会话"""
    # with tf.Session() as sess:
    #     i = 0
    #     # 控制，这次读取到什么地方，下次从什么地方开始读取
    #     # tensorflow提供两个函数
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     try:
    #         while not coord.should_stop() and i < 1:
    #             img, label = sess.run([train_batch, label_batch])
    #             print(img.shape)
    #             if visualize == True:
    #                 for j in np.arange(batch_size):
    #                     plt.figure(j)
    #                     print("label: %d" % label[j])
    #                     plt.imshow(img[j, :, :, 0], cmap='gray')  # img是4D的tensor
    #                     plt.show()
    #             elif visualize == False:
    #                 for j in np.arange(batch_size):
    #                     plt.figure(j)
    #                     print("label: %d" % label[j])
    #             else:
    #                 print('{:-<70}'.format('【*Unknown error!*】'))
    #             i += 1
    #     except tf.errors.OutOfRangeError:
    #         print("done!")
    #     finally:
    #         coord.request_stop()
    #     coord.join(threads)

    print('{:*^70}'.format('【End testing!】'))