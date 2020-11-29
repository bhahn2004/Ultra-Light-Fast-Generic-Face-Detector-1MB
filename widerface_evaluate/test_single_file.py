#!/usr/bin/ python3
# -*- coding: utf-8 -*-
# @Time    : 2019-10-17
# @Author  : vealocia
# @FileName: evaluation_on_widerface.py

import math
import os
import sys

import cv2
sys.path.append('../')
from vision.ssd.config.fd_config import define_img_size

input_img_size = 320  # define input size ,default optional(128/160/320/480/640/1280)
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

label_path = "../models/voc-model-labels.txt"

# net_type = "slim"          # inference faster,lower precision
net_type = "RFB"  # inference lower,higher precision

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = "cuda:0"
# test_device = "cpu"
candidate_size = 800
threshold = 0.1

val_image_root = "/home/bhahn221/Work/protopia/face-detection/Ultra-Light-Fast-Generic-Face-Detector-1MB/data/wider_face/WIDER_val/images"  # path to widerface valuation image root
val_result_txt_save_root = "./widerface_evaluation/"  # result directory

if net_type == 'slim':
    model_path = "../models/pretrained/version-slim-320.pth"
    # model_path = "../models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = "../models/pretrained/version-RFB-320.pth"
    # model_path = "../models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

counter = 0
file_path = '/home/bhahn221/Work/protopia/face-detection/Ultra-Light-Fast-Generic-Face-Detector-1MB/data/wider_face_add_lm_10_10/JPEGImages/0--Parade_0_Parade_marchingband_1_465.jpg'
im = cv2.imread(file_path, cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
print(im.shape)
boxes, labels, probs = predictor.predict(im, candidate_size / 2, threshold)

for i in range(boxes.size(0)):
    bbox = boxes[i, :]
    print('%d %d %d %d %.03f' % (math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1]), probs[i] if probs[i] <= 1 else 1))

# note: with score_threshold = 0.11 and hard_nms, MAP of 320-input model on widerface val set is: 0.785/0.695/0.431
