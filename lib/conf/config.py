# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C


# # Root directory of project
# __C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# # Data directory
# __C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# # Model directory
# __C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models', 'pascal_voc'))


# Use GPU implementation of non-maximum suppression
__C.GUP_MODE = True

# Default GPU device id
__C.GPU_ID = 0


#
# for VOC dataset
#

__C.VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
                    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                    "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

# __C.VOC_CLASSES = [ "bird", "boat", "bottle", "chair", "person", "pottedplant"]
                  
# for SSD, 300*300 or 500*500
__C.VOC_IMAGE_SIZE = 300
# the directory to store the result txt file for each category,
# BASED ON THE DATA LAYER
size_extn = '{}x{}'.format(__C.VOC_IMAGE_SIZE, __C.VOC_IMAGE_SIZE)

__C.VOC_CONF_THRESH = 0.01 # 0.01
__C.VOC_NMS_THRESH = 0.5
__C.VOC_IMG_MEAN_PIXEL = [104,117,123]
# BASED ON THE ROOT DIRECTORY
__C.MODEL_DEF = 'deploy_conv7.prototxt'
__C.MODEL_DIR = 'models/VGGNet/VOC0712/SSD_{}'.format(size_extn)
__C.WEIGHTS_DIR = 'models/VGGNet/VOC0712/SSD_{}'.format(size_extn)
__C.WEIGHTS_DEF = 'VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
# __C.MAX_BOX_SIZE = [32 , 64 , 96 , 128 , 160 , 192 , 224 , 256 , 288 , 300,360,420,500]
# __C.MIN_BOX_SIZE = [1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1, 1,1,1]
__C.MAX_BOX_SIZE = [256,]
__C.MIN_BOX_SIZE = [1,]
model_def_name = __C.MODEL_DEF.split('.')[-2]
__C.VOC_RES_DIR_NAME = 'ssd_{}_{}'.format(size_extn,model_def_name)
__C.VOC_RES_TXT_EXT = 'det_VOC'
__C.VOC_RES_PKL_NAME = 'detections'
#
# Training options
#

__C.TRAIN = edict()

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)
# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128



#
# Testing options
#

__C.TEST = edict()

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

__C.TEST.CONF_FILTER = 0.01
#
# MISC
#





def get_output_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

