import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running_2023 import run_dataset
from lib.test.evaluation.tracker import Tracker

dataset_name = 'vot23'
tracker_name = 'procontext'
tracker_param = 'procontext'
run_ids = 2

dataset = get_dataset(dataset_name)
# # # print(len(dataset[0].ground_truth_rect))
# trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id) for run_id in range(run_ids)]
# print(trackers)
run_dataset(dataset, debug=True, threads=0)