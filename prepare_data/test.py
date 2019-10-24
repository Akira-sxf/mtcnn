# coding:utf-8
import os
import random
import sys
import time

import tensorflow as tf

import sys

p_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(p_dir)
import prepare_data
from prepare_data.tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple

with tf.python_io.TFRecordWriter('../data/imglists/PNet/test.tfrecord') as tfrecord_writer:
	tfrecord_file = '../data/imglists/PNet/train_PNet_landmark.tfrecord_shuffle'
	filename_queue = tf.train.string_input_producer([tfrecord_file])
	# read tfrecord
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	image_features = tf.parse_single_example(
		serialized_example,
		features={
			'image/encoded': tf.FixedLenFeature([], tf.string),  # one image  one record
			'image/label': tf.FixedLenFeature([], tf.int64),
			'image/roi': tf.FixedLenFeature([4], tf.float32),
			'image/landmark': tf.FixedLenFeature([10], tf.float32)
		}
	)
	print(image_features['image/label'])
	# image_features.SerializeToString()
	tfrecord_writer.write(image_features.SerializeToString())
print('finished\n')