import os
import sys
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

prj_path = os.path.join(os.path.abspath(''))
if prj_path not in sys.path:
    sys.path.append(prj_path)

# tracker
import importlib
from pathlib import Path
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running_2023 import run_dataset
from lib.test.evaluation import Sequence, Tracker

import sys
from sam_deaot.SegTracker import SegTracker
from sam_deaot.model_args import aot_args, sam_args, segtracker_args
from sam_deaot.aot_tracker import _palette
from scipy.ndimage import binary_dilation

input_video_path = 'data/vot2023'
folder_name = os.listdir(input_video_path)
folder_name = 'bus'
io_args = {
    'input_img': f'data/vot2023/{folder_name}/color',
    'output_mask_dir': f'output/test/pro_sam_aot/{folder_name}/mask',  # save pred masks
    'output_video': f'output/test/pro_sam_aot/{folder_name}/{folder_name}.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
}

# choose good parameters in sam_args based on the first frame segmentation result
# other arguments can be modified in model_args.py
# note the object number limit is 255 by default, which requires < 10GB GPU memory with amp
sam_args['generator_args'] = {
        'points_per_side': 30,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    }

print(os.environ)
segtracker = SegTracker(io_args, segtracker_args,sam_args,aot_args)
segtracker.restart_tracker()

