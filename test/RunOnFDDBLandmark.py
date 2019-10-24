#coding:utf-8
import sys
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

sys.path.append("..")
import argparse
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
import cv2
import os
data_dir = '../data/FDDB'
out_dir = '../data/FDDB_OUTPUT'

def get_imdb_fddb(data_dir):
    imdb = []
    nfold = 10
    for n in range(nfold):
        file_name = 'FDDB-folds/FDDB-fold-%02d.txt' % (n + 1)
        file_name = os.path.join(data_dir, file_name).replace('\\', '/')
        fid = open(file_name, 'r')
        image_names = []
        for im_name in fid.readlines():
            image_names.append(im_name.strip('\n'))      
        imdb.append(image_names)
    return imdb        



if __name__ == "__main__":
    test_mode = "ONet"
    thresh = [0.6,0.35,0.01]
    min_face_size = 20
    stride = 2
    slide_window = False
    shuffle = False
    vis = False
    detectors = [None, None, None]
    prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
    epoch = [22, 14, 20]
    batch_size = [2048, 256, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    
    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet
    
    # load onet model
    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet
    
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)
    
    
    
    imdb = get_imdb_fddb(data_dir)
    nfold = len(imdb)    
    for i in range(nfold):
        image_names = imdb[i]
        print(image_names)
        dets_file_name = os.path.join(out_dir, 'FDDB-det-fold-%02d.txt' % (i + 1)).replace('\\', '/')
        landmarks_file_name = os.path.join(out_dir, 'FDDB-landmark-fold-%02d.txt' % (i + 1)).replace('\\', '/')
        fid = open(dets_file_name,'w')
        fil = open(landmarks_file_name, 'w')
        sys.stdout.write('%s ' % (i + 1))
        image_names_abs = [os.path.join(data_dir,'originalPics',image_name+'.jpg').replace('\\', '/') for image_name in image_names]
        test_data = TestLoader(image_names_abs)
        all_boxes, all_landmarks = mtcnn_detector.detect_face(test_data)
        
        for idx,im_name in enumerate(image_names):
            img_path = os.path.join(data_dir,'originalPics',im_name+'.jpg').replace('\\', '/')
            image = cv2.imread(img_path)
            boxes = all_boxes[idx]
            landmarks = all_landmarks[idx]
            #print(landmarks)
            if boxes is None:
                fid.write(im_name+'\n')
                fid.write(str(1) + '\n')
                fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                fil.write(im_name + '\n')
                fil.write(str(1) + '\n')
                fil.write('%f %f %f %f %f %f %f %f %f %f\n' % (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
                continue
            fid.write(im_name+'\n')
            fid.write(str(len(boxes)) + '\n')
            fil.write(im_name+'\n')
            fil.write(str(len(boxes)) + '\n')
            
            for box in boxes:
                fid.write('%f %f %f %f %f\n' % (float(box[0]), float(box[1]), float(box[2]-box[0]+1), float(box[3]-box[1]+1),box[4]))
            for land in landmarks:
                fil.write('%f %f %f %f %f %f %f %f %f %f\n' % (float(land[0]), float(land[1]), float(land[2]), float(land[3]), float(land[4]), float(land[5]), float(land[6]), float(land[7]), float(land[8]), float(land[9])))
                
        fid.close()
