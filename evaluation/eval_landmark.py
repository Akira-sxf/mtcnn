import sys
sys.path.append("..")
import numpy as np
import argparse
import math
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
import cv2
import os

def dist_s(a1, a2, t):
	res = 0
	for i in range(10):
		res += pow(a1[i] - a2[i + 4], 2)
	if t == 1:
		res  = res / (5.0 * pow(pow(a2[4] - a2[6], 2) + pow(a2[5] - a2[7], 2), 1))
	elif t == 0:
		res = res / (5.0 * a2[2] * a2[3])
	return res

land_dir = '../data/FDDB_OUTPUT'
land_dir = '../data/FDDB_OUTPUT_ORI'
anno_dir = '../data/FDDB_annotation'

land_f = open(os.path.join(land_dir, 'FDDB-landmark-fold-all.txt').replace('\\', '/'), 'r')
anno_f = open(os.path.join(anno_dir, 'FDDB-landmark-fold-all.txt').replace('\\', '/'), 'r')
lines_land = land_f.readlines()
lines_anno = anno_f.readlines()
land_f.close()
anno_f.close()
i = 1
j = 1
faces = 0
anno_faces = 0
sum = 0
images = 0
normalize_type = 0 # 0 - face size, 1 - inter-pupil distance
while i < len(lines_land) and j < len(lines_anno):
	if images % 100 == 0:
		print('images : ', images)
		print('sum : ', sum)
	images += 1

	land_num = int(lines_land[i])
	anno_num = min(5, int(lines_anno[j]))
	anno_faces += anno_num
	if anno_num == 0:
		i += land_num + 2
		j += anno_num + 2
		continue
	faces += land_num
	temp1 = []
	temp2 = []
	for m in range(i + 1, i + 1 + land_num):
		a = lines_land[m].strip().split()
		# print(a)
		tt = [float(k) for k in a]
		temp1.append(tt)
	for n in range(j + 1, j + 1 + anno_num):
		a = lines_anno[n].strip().split()
		# print(a)
		tt = [float(k) for k in a]
		temp2.append(tt)
	temp1 = np.array(temp1)
	temp2 = np.array(temp2)
	for p in range(land_num):
		# if anno_num == 0:
			# sum += dist_s(temp1[p], np.zeros((10)))
			# continue
		min_sqr = dist_s(temp1[p], temp2[0], normalize_type)
		for q in range(1, anno_num):
			min_sqr = min(min_sqr, dist_s(temp1[p], temp2[q], normalize_type))
		sum += min_sqr

	i += land_num + 2
	j += anno_num + 2

sum = sum / faces
print('faces : ', faces)
print('anno_faces : ', anno_faces)
print('landmark mean error : ', sum)

total = 0
f = open('../data/FDDB/FDDB-folds/FDDB-fold-all-ellipseList.txt', 'r')
line = f.readline()
while line:
	if line.find('big') > 0:
		line = f.readline()
		total += int(line)
	line = f.readline()
f.close()
print('total faces : ', total)