import os
import re
from shutil import copyfile

LIST = '/home/bhahn221/Work/protopia/face-detection/Ultra-Light-Fast-Generic-Face-Detector-1MB/data/wider_face_add_lm_10_10/ImageSets/Main/test (copy).txt'
SRC = '/home/bhahn221/Work/protopia/face-detection/Ultra-Light-Fast-Generic-Face-Detector-1MB/data/wider_face/WIDER_val/images/'
DST = '/home/bhahn221/Work/protopia/face-detection/Ultra-Light-Fast-Generic-Face-Detector-1MB/data/wider_face_add_lm_10_10/JPEGImages/'

with open(LIST, 'r') as f:
    filenames = [filename.rstrip() for filename in f.readlines()]

for filename in filenames:
    filename_split = re.split(r'(\d+)', filename)
    filename_split.remove('')
    
    imgdir = ''.join(filename_split[:2]).rstrip('_')
    imgname = ''.join(filename_split[2:-1])

    src = SRC+imgdir+'/'+imgname+'.jpg'
    dst = DST+filename+'.jpg'

    print(f'COPY {src} {dst}')

    copyfile(src, dst)
