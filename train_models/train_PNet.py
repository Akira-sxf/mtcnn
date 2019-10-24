#coding:utf-8
import sys
sys.path.insert(0,'..')
from train_models.mtcnn_model import P_Net
from train_models.train import train

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train_PNet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch: max epoch for training
    :param display:
    :param lr: learning rate
    :return:
    """
    net_factory = P_Net
    train(net_factory,prefix, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    #data path
    base_dir = '../data/imglists/PNet'
    model_name = 'MTCNN'
    #model_path = '../data/%s_model/PNet/PNet' % model_name
    #with landmark
    model_path = '../data/%s_model/PNet_landmark/PNet' % model_name
            
    prefix = model_path
    end_epoch = 30
    display = 100
    lr = 0.001
    train_PNet(base_dir, prefix, end_epoch, display, lr)
