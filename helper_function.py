import os
import numpy as np
import cv2
import torch
import gc
from scipy.ndimage import binary_dilation
from aot_tracker import _palette
from PIL import Image
import glob

def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))

def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)
 
    return mask, (tl_x, tl_y)

def read_label(input_video_path, number_objects):
    
    folder_name = os.path.split(input_video_path)[0]
    tmp_path = folder_name + '/color/'
    image_path = os.path.join(tmp_path,os.listdir(tmp_path)[0])
    img = cv2.imread(image_path)
    height, width, channel = img.shape
    f = open(folder_name + '/groundtruth.txt', "r")
    
    squeezed_label = np.zeros((height,width)) 
    
    first_mask = 0 
    print(type(f))
    for x in f:
        first_mask = x[1:]
        break
    label, (x, y) = create_mask_from_string(list(map(lambda x: int(x), first_mask.split(","))))
    h,w = label.shape

    for obj_id in range(1,number_objects+1):
        squeezed_label[y:y+h, x:x+w] += (label * obj_id).astype(np.uint8)
    return squeezed_label.astype(np.uint8)
    

def read_multi_object(input_video_path):
    input = os.path.split(input_video_path)[0]
    tmp_path = input + '/color/'
    image_path = os.path.join(tmp_path,os.listdir(tmp_path)[0])
    img = cv2.imread(image_path)
    height, width, channel = img.shape
    squeezed_label = np.zeros((height,width)) 
    
    path = f'{input}/groundtruth_*.txt'
    files = glob.glob(path)
    if len(files) == 0:
        path = f'{input}/groundtruth.txt'
        files = glob.glob(path)
    mask_list = []
    obj_id = 0
    for file in files:
        obj_id += 1
        f = open(file, "r")
        for x in f:
            mask_list.append(x[1:])
            break
        label, (x, y) = create_mask_from_string(list(map(lambda x: int(x), mask_list[-1].split(","))))
        h,w = label.shape
        squeezed_label[y:y+h, x:x+w] += (label * obj_id).astype(np.uint8)
    return squeezed_label


def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)


def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def read_groundtruth(input_video_path, number_objects):
    
    folder_name = os.path.split(input_video_path)[0]
    tmp_path = folder_name + '/color/'
    image_path = os.path.join(tmp_path,os.listdir(tmp_path)[0])
    img = cv2.imread(image_path)
    height, width, channel = img.shape
    f = open(folder_name + '/groundtruth.txt', "r")
    mask = [] 
    pred_list = []
    for x in f:
        mask.append(x[1:])
        label, (x, y) = create_mask_from_string(list(map(lambda x: int(x), mask[-1].split(","))))
        h,w = label.shape
        squeezed_label = np.zeros((height,width)) 
        for obj_id in range(1,number_objects+1):
            squeezed_label[y:y+h, x:x+w] += (label * obj_id).astype(np.uint8)
            pred_list.append(squeezed_label)
        
            
    folder_name = 'bag'
    io_args = {
        'input_video': f'./VOTS/sequences/{folder_name}/{folder_name}.mp4',
        'output_mask_dir': f'./problem_solving_output/{folder_name}', # save pred masks
        'output_video': f'./problem_solving_output/{folder_name}.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
    }

    # draw pred mask on frame and save as a video
    cap = cv2.VideoCapture(io_args['input_video'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if io_args['input_video'][-3:]=='mp4':
        fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    elif io_args['input_video'][-3:] == 'avi':
        fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
        # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame,pred_mask)
        # masked_frame = masked_pred_list[frame_idx]
        masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(frame_idx),end='\r')
        frame_idx += 1
    out.release()
    cap.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')
    # manually release memory (after cuda out of memory)
    torch.cuda.empty_cache()
    gc.collect()
    
