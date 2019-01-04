from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.DATASET = edict()

__C.DATASET.version = '2017_11'

__C.TRAIN = edict()

__C.TRAIN.snapshot = None  # Resume training from a snapshot

__C.TRAIN.imagenet_weights = None

__C.TRAIN.weights = '/home/yul-j/Desktop/Safetyhat/safetyhat-retinanet/weights/ResNet-50-model.keras.h5'

__C.TRAIN.no_weights = False

__C.TRAIN.backbone = 'resnet50'

__C.TRAIN.batch_size = 4

__C.TRAIN.gpu = 0

__C.TRAIN.multi_gpu = 1

__C.TRAIN.multi_gpu_force = False

__C.TRAIN.evaluation = True

__C.TRAIN.freeze_backbone = True

__C.TRAIN.image_min_side = 800

__C.TRAIN.image_max_side = 1080

__C.TRAIN.save_path = None

__C.TRAIN.sigma = 3  # Parameter for smooth l1 loss

__C.TRAIN.translation_min = (0.0, 0.0)

__C.TRAIN.translation_max = (0.0, 0.0)
