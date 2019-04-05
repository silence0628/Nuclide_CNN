# -*- coding: utf-8 -*-
"""
    nuclide identification
    improve code
    update: 2019.04.03
"""

import argparse
from glob import glob
import os
import tensorflow as tf
import numpy as np
from utils import get_train_files, get_train_batch, get_test_files, get_test_batch
from model import inference, inference1D, losses, trainning, evaluation
from PIL import Image
import scipy.io as scio
import time

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=20, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint/', help='models are saved here')
parser.add_argument('--train_dir', dest='train_dir', default='/home/sjj/Spectra_train/', help='data set for training')
parser.add_argument('--test_dir', dest='test_dir', default='./test/', help='data set for testing')
parser.add_argument('--iteration', dest='iters', type=int, default=600, help='max iteration for training')
parser.add_argument('--aim_layer', dest='aim_layer', type=int, default=6, help='ectract aim_layer feature')
parser.add_argument('--feature_dir', dest='feature_dir', default='./save_features/', help='dir for saving features')
parser.add_argument('--inference', dest='inference', default='2D', help='true for 2D model, false for 1D model')
parser.add_argument('--n_class', dest='n_class', type=int, default=6, help='category of aim nuclides')
parser.add_argument('--isSaveFeature', dest='isSaveFeature', default=True, help='save or not save features')
args = parser.parse_args()


def run_train(image_batch, label_batch, n_class, batch_size, checkpoint_dir, lr, MAX_STEP):
    print('{:*^70}'.format('【train starting!】'))
    global result
    if args.inference == '2D':
        result = inference(image_batch, batch_size, n_class)
    elif args.inference == '1D':
        result = inference1D(image_batch, batch_size, n_class)
    else:
        print('【Unknown error】')

    print('{:*^70}'.format('DEBUG'))
    train_loss = losses(result[-1], label_batch)
    train_op = trainning(train_loss, lr)
    train__acc = evaluation(result[-1], label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

    print('{:*^70}'.format(' train ending! '))

def run_test(test_batch, n_class, checkpoint_dir, batch_size):
    """批量读取待测样本并进行提取数据"""
    print('{:*^70}'.format('【test  starting!】'))
    print(test_batch.shape)
    test_batch = tf.reshape(test_batch, (20, 208, 208, 3))
    print(test_batch.shape)
    logits = inference(test_batch, batch_size, n_class)
    result = tf.nn.softmax(logits)
    x = tf.placeholder(tf.float32, shape=[20, 208, 208, 3])

    saver = tf.train.Saver()

    with tf.Session() as sess:
        i = 0
        # 控制，这次读取到什么地方，下次从什么地方开始读取
        # tensorflow提供两个函数
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            ckpt_name = ckpt.model_checkpoint_path.split('/')[-1]
            # saver.restore(sess, ckpt_name)  # windows 环境下
            saver.restore(sess, ckpt.model_checkpoint_path)  #  ubuntu环境下
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
        try:
            while not coord.should_stop() and i < 1:
                img = sess.run(test_batch)
                print(img.shape)
                """输出可视化的各层特征"""
                middle= sess.run(result, feed_dict={x: img})
                print(middle)
                i+=1
        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
        coord.join(threads)

    print('{:*^70}'.format('【test ending !】'))

def feature_extraction(test_batch, n_class, checkpoint_dir, batch_size):
    """批量读取待测样本并进行提取数据
        aim_layer: 0 ---> conv1
                   1 ---> pool1
                   2 ---> conv2
                   3 ---> pool2
                   4 ---> local3
                   5 ---> local4
                  -1 ---> softmax
    """
    print('{:*<70}'.format('【feature_extraction  starting!】'))
    global result
    if args.inference == '2D':
        test_batch = tf.reshape(test_batch, (args.batch_size, 32, 32, 1))
        result = inference(test_batch, batch_size, n_class)
        logits = tf.nn.softmax(result[-1])
        x = tf.placeholder(tf.float32, shape=[args.batch_size, 32, 32, 1])

    elif args.inference == '1D':
        test_batch = tf.reshape(test_batch, (args.batch_size, 1, 1024, 1))
        result = inference1D(test_batch, batch_size, n_class)
        logits = tf.nn.softmax(result[-1])
        x = tf.placeholder(tf.float32, shape=[args.batch_size, 1, 1024, 1])

    else:
        print('【Unknown error!】')


    saver = tf.train.Saver()

    with tf.Session() as sess:
        i = 0
        # 控制，这次读取到什么地方，下次从什么地方开始读取
        # tensorflow提供两个函数
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            ckpt_name = ckpt.model_checkpoint_path.split('/')[-1]
            # saver.restore(sess, ckpt_name)  # windows 环境下
            saver.restore(sess, ckpt.model_checkpoint_path)  #  ubuntu环境下
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
        try:
            while not coord.should_stop() and i < 1:
                img = sess.run(test_batch)
                """输出可视化的各层特征"""
                res = sess.run(result, feed_dict={x: img})
                print(res)
                i += 1
        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
        coord.join(threads)

    print('{:*<70}'.format(str(res[0].shape)))
    print('{:*<70}'.format(str(res[1].shape)))
    print('{:*<70}'.format(str(res[2].shape)))
    print('{:*<70}'.format(str(res[3].shape)))
    print('{:*<70}'.format(str(res[4].shape)))
    print('{:*<70}'.format(str(res[5].shape)))
    print('{:*<70}'.format(str(res[6].shape)))



    strs = '_{}_'.format(args.n_class) + time.strftime("%m%d%H%M", time.localtime())
    if args.isSaveFeature == True:
        print('{:*<70}'.format('【Start Saving Features Mat!】'))

        scio.savemat(args.feature_dir + 'SixSpectraAimLayer' + strs + '.mat', {'conv1': res[0],
                    'pool1': res[1], 'conv2': res[2], 'pool2': res[3], 'local3': res[4],
                    'local4': res[5], 'softmax': res[6]})
        print('{:*<70}'.format('【feature_extraction ending !】'))
    else:
        print('{:*<70}'.format('【Not Saving Features Mat!】'))
        print('{:*<70}'.format('【feature_extraction ending !】'))



def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)


    IMG_W = 32
    IMG_H = 32
    CAPACITY = 256

    if args.phase == 'train':

        image_list, label_list = get_train_files(args.train_dir)
        train_batch, label_batch = get_train_batch(image_list, label_list, IMG_W, IMG_H, args.batch_size, CAPACITY)
        run_train(image_batch=train_batch, label_batch=label_batch, n_class=args.n_class,
                  batch_size=args.batch_size, checkpoint_dir=args.ckpt_dir, lr=args.lr, MAX_STEP=args.iters)

    elif args.phase == 'test':

        test_list = get_test_files(args.test_dir)
        test_batch = get_test_batch(test_list, IMG_W, IMG_H, args.batch_size, CAPACITY)
        print(test_batch)
        run_test(test_batch, args.n_class, args.ckpt_dir, args.batch_size)

    elif args.phase == 'feature_extraction':

        image_list, label_list = get_train_files(args.train_dir)
        train_batch, label_batch = get_train_batch(image_list, label_list, IMG_W, IMG_H, args.batch_size, CAPACITY)
        feature_extraction(train_batch, args.n_class, args.ckpt_dir, args.batch_size)

    else:
        print('{:*^50}'.format('【Unknown phase!】'))



if __name__=='__main__':
    tf.app.run()