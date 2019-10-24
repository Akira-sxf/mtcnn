# coding: utf-8
import os
import random
from os.path import join, exists

import cv2
import numpy as np
import numpy.random as npr

import sys
p_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(p_dir)
import prepare_data
from prepare_data.BBox_utils import getDataFromTxt2, BBox
from prepare_data.Landmark_utils import rotate, flip
from prepare_data.utils import IoU




def GenerateData(ftxt,data_path,net,argument=False):
    '''

    :param ftxt: name/path of the text file that contains image path,
                bounding box, and landmarks

    :param output: path of the output dir
    :param net: one of the net in the cascaded networks
    :param argument: apply augmentation or not
    :return:  images and related landmarks
    '''
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
    #
    f = open(join(OUTPUT,"landmark_%s_aug_wider.txt" %(size)).replace('\\','/'),'w') #add img
    #dstdir = "train_landmark_few"
    # get image path , bounding box, and landmarks from file 'ftxt'
    data = getDataFromTxt2(ftxt,data_path=data_path)
    idx = 0
    #image_path bbox landmark(5*2)
    for (imgPath, bbox, landmarkGt) in data:
        #print imgPath
        F_imgs = []
        F_landmarks = []
        #print(imgPath)
        img = cv2.imread(imgPath)

        if image_id == 0:
            print(imgPath)
        assert(img is not None)
        img_h,img_w,img_c = img.shape
        gt_box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])
        #get sub-image from bbox
        f_face = img[bbox.top:(bbox.bottom+1), bbox.left:(bbox.right+1)]
        # print(bbox.left)
        # print(bbox.right)
        # print(bbox.top)
        # print(bbox.bottom)
        # print(img.shape)
        # print(img)
        # print(f_face)
        # resize the gt image to specified size
        f_face = cv2.resize(f_face,(size,size))
        #initialize the landmark
        landmark = np.zeros((5, 2))

        #normalize land mark by dividing the width and height of the ground truth bounding box
        # landmakrGt is a list of tuples
        for index, one in enumerate(landmarkGt):
            # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
            rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            # put the normalized value into the new list landmark
            landmark[index] = rv
        
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        landmark = np.zeros((5, 2)) 
                    
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
        #print F_imgs.shape
        #print F_landmarks.shape
        for i in range(len(F_imgs)):
            if image_id % 100 == 0:
                print("image id : ", image_id)
            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                continue

            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                continue

            cv2.imwrite(join(dstdir,"%d.jpg" %(image_id)).replace('\\','/'), F_imgs[i])
            landmarks = map(str,list(F_landmarks[i]))
            f.write(join(dstdir,"wider_%d.jpg" %(image_id)).replace('\\','/')+" -2 "+" ".join(landmarks)+"\n")
            image_id = image_id + 1

    print('total images : ', image_id)
    f.close()
    return F_imgs,F_landmarks

if __name__ == '__main__':
    dstdir = "../data/12/train_PNet_landmark_aug_wider"
    OUTPUT = '../data/imglists/PNet'
    data_path = '../data/WIDER_train/images'
    if not exists(OUTPUT):
        os.mkdir(OUTPUT)
    if not exists(dstdir):
        os.mkdir(dstdir)
    assert (exists(dstdir) and exists(OUTPUT))
    # train data
    net = "PNet"
    #the file contains the names of all the landmark training data
    train_txt = "Wider_label.txt"
    imgs,landmarks = GenerateData(train_txt,data_path,net)
