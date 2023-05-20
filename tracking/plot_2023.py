import os
import glob
import cv2
import argparse
from random import randint

default_output_path = "/home/son/Desktop/vot/vot2023/output/test/tracking_results/procontext/{}"
default_data_path = "/home/son/Desktop/vot/vot2023/data/vot2023/{}/color/*"

def run(seq_name):   
    seq_output_path = default_output_path.format(seq_name)
    seq_data_path = default_data_path.format(seq_name)
    vid_out = seq_output_path + '/output.avi'

    objs = len(os.listdir(seq_output_path))
    gt = {}
    colors = {}

    for obj_id in os.listdir(seq_output_path):
        bbox = []
        gt_path = os.path.join(seq_output_path, obj_id, seq_name + ".txt")
        with open(gt_path, 'r') as file:
            for line in file:
                bbox.append([int(x) for x in line.rstrip().split()])
        obj_id = int(obj_id)
        gt[obj_id - 1] = bbox

    imgs = sorted(glob.glob(seq_data_path))
    frame = cv2.imread(imgs[0])
    img_h, img_w, channels = frame.shape

    video = cv2.VideoWriter(vid_out, 0, 10, (img_w, img_h))

    for frame_id, img in enumerate(imgs):
        frame2 = cv2.imread(img)
        for obj in range(objs):
            if obj not in colors:
                colors[obj] = [randint(0, 255), randint(0, 255), randint(0, 255)]

            x, y, w, h = gt[obj][frame_id]
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            
            cv2.rectangle(frame2, (x_min, y_min), (x_max, y_max), color=colors[obj], thickness=2)
        video.write(frame2)

    cv2.destroyAllWindows()
    video.release()

def main():
    parser = argparse.ArgumentParser(description='Plot multiple object with output of ProContEXT')
    parser.add_argument('--sequence_name', type=str, default="*", help='Sequence name to plot')

    args = parser.parse_args()

    if args.sequence_name == "*":
        seq_dir = default_output_path.format("*")
        seq_path_list = [os.path.split(seq_path)[-1] for seq_path in glob.glob(seq_dir)]
        for seq in seq_path_list:
            print("[INFO] Processing: {} sequence".format(seq))
            run(seq)
    else:
        run(args.sequence_name)


if __name__ == '__main__':
    main()