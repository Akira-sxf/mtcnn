# coding: utf-8
import os
import random
from os.path import join, exists

import cv2
import numpy as np
import numpy.random as npr
import sys
sys.path.insert(0,'..')
from prepare_data.BBox_utils import getDataFromTxt2, BBox
from prepare_data.Landmark_utils import rotate, flip

dstdir = "../data/48/train_ONet_landmark_aug_wider"
OUTPUT = '../data/imglists_noLM/ONet'
if not exists(OUTPUT): os.mkdir(OUTPUT)
if not exists(dstdir): os.mkdir(dstdir)
assert(exists(dstdir) and exists(OUTPUT))

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
     # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter*1.0 / (box_area + area - inter)
    return ovr
def GenerateData(ftxt, data_path, net, argument=False):
    if net == "PNet":
        size = 12
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48
    else:
        print('Net type error')
        return
    image_id = 0
    f = open(join(OUTPUT,"landmark_%s_aug_wider.txt" %(size)).replace('\\', '/'),'a')
    data = getDataFromTxt2(ftxt, data_path)
    idx = 0
    #image_path bbox landmark(5*2)
    for (imgPath, bbox, landmarkGt) in data:
        #print imgPath
        F_imgs = []
        F_landmarks = []        
        img = cv2.imread(imgPath)
        assert(img is not None)
        img_h,img_w,img_c = img.shape
        gt_box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])
        f_face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
        f_face = cv2.resize(f_face,(size,size))
        landmark = np.zeros((5, 2))
        #normalize
        for index, one in enumerate(landmarkGt):
            rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            landmark[index] = rv
        
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        landmark = np.zeros((5, 2))        

        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
        #print F_imgs.shape
        #print F_landmarks.shape
        for i in range(len(F_imgs)):
            if image_id % 100 == 0:
                sys.stdout.write('\r>> image_id : %d' % image_id)
            sys.stdout.flush()

            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                continue

            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                continue

            cv2.imwrite(join(dstdir,"%d.jpg" %(image_id)).replace('\\','/'), F_imgs[i])
            landmarks = map(str,list(F_landmarks[i]))
            f.write(join(dstdir,"%d.jpg" %(image_id)).replace('\\','/')+" -2 "+" ".join(landmarks)+"\n")
            image_id = image_id + 1
            
    #print F_imgs.shape
    #print F_landmarks.shape
    #F_imgs = processImage(F_imgs)
    #shuffle_in_unison_scary(F_imgs, F_landmarks)
    
    f.close()
    return F_imgs,F_landmarks



if __name__ == '__main__':
    # train data
    net = "ONet"
    #train_txt = "train.txt"
    train_txt = "Wider_label.txt"
    data_path = '../data/WIDER_train/images'
    imgs,landmarks = GenerateData(train_txt, data_path, net)
    #WriteToTfrecord(imgs,landmarks,net)
   
