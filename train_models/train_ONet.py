#coding:utf-8
import sys
sys.path.insert(0,'..')
from train_models.mtcnn_model import O_Net
from train_models.train import train

import os
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train_ONet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = O_Net
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    base_dir = '../data/imglists_noLM/ONet'

    model_name = 'MTCNN'
    model_path = '../data/%s_model/ONet_landmark/ONet' % model_name
    prefix = model_path
    end_epoch = 22
    display = 10
    lr = 0.001
    train_ONet(base_dir, prefix, end_epoch, display, lr)
