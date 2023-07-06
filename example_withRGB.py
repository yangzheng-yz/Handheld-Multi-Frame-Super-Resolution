# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:10:18 2023

This is an example script showing how to call the pipeline and specify
the parameters.

Make sure that the example bursts have been downloaded from the release version!!


@author: jamyl
"""

import os
import matplotlib.pyplot as plt
from handheld_super_resolution.super_resolution import processRGB
from skimage import img_as_ubyte
import numpy as np
import pickle as pkl
import cv2 as cv
import torch
from camera_pipeline import apply_gains, apply_ccm, apply_smoothstep, gamma_compression


def torch_to_numpy(a: torch.Tensor):
    return a.permute(1, 2, 0).numpy()

def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1)

def torch_to_npimage(a: torch.Tensor, unnormalize=True, input_bgr=False):
    if isinstance(a, torch.Tensor):
        a_np = torch_to_numpy(a.clamp(0.0, 1.0))

    if unnormalize:
        a_np = a_np * 255
    a_np = a_np.astype(np.uint8)

    if input_bgr:
        return a_np

    return cv.cvtColor(a_np, cv.COLOR_RGB2BGR)

def process_linear_image_rgb(image, meta_info, gains=True, ccm=True, gamma=True, smoothstep=True, return_np=False):
    if not isinstance(image, torch.Tensor):
        image = numpy_to_torch(image)
    if gains:
        image = apply_gains(image, meta_info['rgb_gain'], meta_info['red_gain'], meta_info['blue_gain'])

    if ccm:
        image = apply_ccm(image, meta_info['cam2rgb'])

    image = image.clamp(0.0, 1.0)
    if meta_info['gamma'] and gamma:
        image = gamma_compression(image)

    if meta_info['smoothstep'] and smoothstep:
        image = apply_smoothstep(image)

    image = image.clamp(0.0, 1.0)

    if return_np:
        image = torch_to_npimage(image)
    return image

def opencv_loader(path):
    """ Read image using opencv's imread function and returns it in rgb format"""
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)

        # convert to rgb and return
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None

# Specify verbose options
options = {'verbose' : 1}



# Specify the scale as follows. All the parameters are automatically 
# choosen but can be overwritten : check params.py to see the entire list
# of configurable parameters.
params={
        "scale":2,
        "merging" : {
            'kernel': 'handheld'},
        'post processing' : {'on':False},
        # 'robustness': {'on':False},
        # 'accumulated robustness denoiser': {'merge': {'on':False}}
        # Post processing is enabled by default,
        # but it can be turned off here
        }

# calling the pipeline
permutation = np.array([
    [0,0],
    [0,1],
    [1,1],
    [1,0]
])

burst_transformation_params = {'max_translation': 3.0,
                                    'max_rotation': 0.0,
                                    'max_shear': 0.0,
                                    'max_scale': 0.0,
                                    'random_pixelshift': True,
                                    'specified_translation': permutation}

with open(os.path.join('/home/yutong/zheng/projects/Handheld-Multi-Frame-Super-Resolution/val_meta_infos.pkl'), 'rb') as f:
    meta_infos_val = pkl.load(f)
image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True, 'predefined_params': meta_infos_val}

burst_size = 14
downsample_factor = params['scale']
crop_sz = (384, 384)

image_folder = '/mnt/data0/zheng/NightCity_1024x512/val'
image_files = os.listdir(image_folder)
for image_file in image_files:
    if image_file != 'Chicago_0006.png':
        continue
    rgb_img = opencv_loader(os.path.join(image_folder, image_file))
    output_img, frame_gt = processRGB(rgb_img, image_name=image_file, burst_transformation_params=burst_transformation_params,
                            image_processing_params=image_processing_params, burst_size=burst_size, 
                            downsample_factor=downsample_factor, crop_sz=crop_sz, options=options, custom_params=params)
    
    frame_gt = process_linear_image_rgb(frame_gt, meta_infos_val['Chicago_0006.png'], return_np=True)
    output_img = process_linear_image_rgb(output_img, meta_infos_val['Chicago_0006.png'], return_np=True)

# saving the result
os.makedirs('./results', exist_ok=True)
frame_gt = frame_gt[:,:,[2,1,0]]
output_img = output_img[:,:,[2,1,0]]
plt.imsave('./results/Chicago_0006.png', img_as_ubyte(output_img))
plt.imsave('./results/Chicago_0006_gt.png', img_as_ubyte(frame_gt))


# # plotting the result
# plt.figure("output")
# plt.imshow(output_img, interpolation = 'none')
# plt.xticks([])
# plt.yticks([])

