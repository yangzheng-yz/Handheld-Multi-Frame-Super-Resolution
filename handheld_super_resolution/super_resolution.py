# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:56:22 2022

This script contains : 
    - The implementation of Alg. 1, the main the body of the method
    - The implementation of Alg. 2, where the function necessary to
        compute the optical flow are called
    - All the operations necessary before its call, such as whitebalance,
        exif reading, and manipulations of the user's parameters.
    - The call to post-processing operations (if enabled)


@author: jamyl
"""

import os
import time
import warnings

from pathlib import Path
import numpy as np
from numba import cuda
import rawpy
import torch
import cv2
import random

from .utils_image import compute_grey_images, frame_count_denoising_gauss, frame_count_denoising_median, apply_orientation
from .utils import getTime, DEFAULT_NUMPY_FLOAT_TYPE, divide, add, round_iso, timer
from .block_matching import init_block_matching, align_image_block_matching
from .params import check_params_validity, get_params, merge_params
from .robustness import init_robustness, compute_robustness
from .utils_dng import load_dng_burst
from .ICA import ICA_optical_flow, init_ICA
from .fast_monte_carlo import run_fast_MC
from .kernels import estimate_kernels
from .merge import merge, merge_ref
from . import raw2rgb
from . import rgb2raw

NOISE_MODEL_PATH = Path(os.path.dirname(__file__)).parent / 'data' 

def torch_to_numpy(a: torch.Tensor):
    return a.permute(1, 2, 0).numpy()

def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1)

def center_crop(frames, crop_sz):
    """
    :param frames: Input frame as tensor (H, W, C)
    :param crop_sz: Output crop sz as (rows, cols)

    :return:
    """
    if not isinstance(crop_sz, (tuple, list)):
        crop_sz = (crop_sz, crop_sz)
        
    shape = frames.shape

    r1 = ((shape[-2] - crop_sz[-2]) // 2)
    c1 = ((shape[-1] - crop_sz[-1]) // 2)

    r2 = r1 + crop_sz[-2]
    c2 = c1 + crop_sz[-1]

    frames_crop = frames[:, r1:r2, c1:c2]

    return frames_crop

def get_tmat(image_shape, translation, theta, shear_values, scale_factors):
    """ Generates a transformation matrix corresponding to the input transformation parameters """
    im_h, im_w = image_shape

    t_mat = np.identity(3)

    t_mat[0, 2] = translation[0]
    t_mat[1, 2] = translation[1]
    t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
    t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

    t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                        [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                        [0.0, 0.0, 1.0]])

    t_scale = np.array([[scale_factors[0], 0.0, 0.0],
                        [0.0, scale_factors[1], 0.0],
                        [0.0, 0.0, 1.0]])

    t_mat = t_scale @ t_rot @ t_shear @ t_mat

    t_mat = t_mat[:2, :]

    return t_mat

def single2lrburstdatabase(image, burst_size, downsample_factor=1, transformation_params=None,
                   interpolation_type='bilinear'):
    """ Generates a burst of size burst_size from the input image by applying random/specific transformations defined by
    transformation_params, and downsampling the resulting burst by downsample_factor.

    args:
        image - input sRGB image
        burst_size - Number of images in the output burst
        downsample_factor - Amount of downsampling of the input sRGB image to generate the LR image
        transformation_params - Parameters of the affine transformation used to generate a burst from single image
        interpolation_type - interpolation operator used when performing affine transformations and downsampling
    """

    if interpolation_type == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif interpolation_type == 'lanczos':
        interpolation = cv2.INTER_LANCZOS4
    else:
        raise ValueError

    normalize = False
    if isinstance(image, torch.Tensor):
        if image.max() < 2.0:
            image = image * 255.0
            normalize = True
        image = torch_to_numpy(image).astype(np.uint8)

    burst = []
    sample_pos_inv_all = []

    rvs, cvs = torch.meshgrid([torch.arange(0, image.shape[0]),
                               torch.arange(0, image.shape[1])])

    sample_grid = torch.stack((cvs, rvs, torch.ones_like(cvs)), dim=-1).float()

    for i in range(burst_size):
        if i == 0:
            # For base image, do not apply any random transformations. We only translate the image to center the
            # sampling grid
            shift = (downsample_factor / 2.0) - 0.5
            translation = (shift, shift)
            theta = 0.0
            shear_factor = (0.0, 0.0)
            scale_factor = (1.0, 1.0)
        else:
            # Sample random image transformation parameters
            if transformation_params.get('random_pixelshift', False):
                max_translation = transformation_params.get('max_translation', 0.0)

                if max_translation <= 0.01:
                    shift = (downsample_factor / 2.0) - 0.5
                    translation = (shift, shift)
                else:
                    translation = (random.uniform(-max_translation, max_translation),
                                random.uniform(-max_translation, max_translation))
                    
                max_rotation = transformation_params.get('max_rotation', 0.0)
                theta = random.uniform(-max_rotation, max_rotation)
            
                max_shear = transformation_params.get('max_shear', 0.0)
                shear_x = random.uniform(-max_shear, max_shear)
                shear_y = random.uniform(-max_shear, max_shear)
                shear_factor = (shear_x, shear_y)

                max_ar_factor = transformation_params.get('max_ar_factor', 0.0)
                ar_factor = np.exp(random.uniform(-max_ar_factor, max_ar_factor))

                max_scale = transformation_params.get('max_scale', 0.0)
                scale_factor = np.exp(random.uniform(-max_scale, max_scale))

                scale_factor = (scale_factor, scale_factor * ar_factor)
            
            else:
                specified_translation = transformation_params.get('specified_translation', np.array([]))
                assert len(specified_translation)==burst_size, "The number of specified translation modes must be equal burst size. "

                translation = (specified_translation[i][0], specified_translation[i][1])
                theta = 0.0
                shear_factor = (0.0, 0.0)
                scale_factor = (1.0, 1.0)

        output_sz = (image.shape[1], image.shape[0])

        # Generate a affine transformation matrix corresponding to the sampled parameters
        t_mat = get_tmat((image.shape[0], image.shape[1]), translation, theta, shear_factor, scale_factor)
        t_mat_tensor = torch.from_numpy(t_mat)

        # Apply the sampled affine transformation
        image_t = cv2.warpAffine(image, t_mat, output_sz, flags=interpolation,
                                 borderMode=cv2.BORDER_CONSTANT)

        t_mat_tensor_3x3 = torch.cat((t_mat_tensor.float(), torch.tensor([0.0, 0.0, 1.0]).view(1, 3)), dim=0)
        t_mat_tensor_inverse = t_mat_tensor_3x3.inverse()[:2, :].contiguous()

        sample_pos_inv = torch.mm(sample_grid.view(-1, 3), t_mat_tensor_inverse.t().float()).view(
            *sample_grid.shape[:2], -1)

        if transformation_params.get('border_crop') is not None:
            border_crop = transformation_params.get('border_crop')

            image_t = image_t[border_crop:-border_crop, border_crop:-border_crop, :]
            sample_pos_inv = sample_pos_inv[border_crop:-border_crop, border_crop:-border_crop, :]

        # Downsample the image
        image_t = cv2.resize(image_t, None, fx=1.0 / downsample_factor, fy=1.0 / downsample_factor,
                             interpolation=interpolation)
        sample_pos_inv = cv2.resize(sample_pos_inv.numpy(), None, fx=1.0 / downsample_factor,
                                    fy=1.0 / downsample_factor,
                                    interpolation=interpolation)

        sample_pos_inv = torch.from_numpy(sample_pos_inv).permute(2, 0, 1)

        if normalize:
            image_t = numpy_to_torch(image_t).float() / 255.0
        else:
            image_t = numpy_to_torch(image_t).float()
        burst.append(image_t)
        sample_pos_inv_all.append(sample_pos_inv / downsample_factor)

    burst_images = torch.stack(burst)
    sample_pos_inv_all = torch.stack(sample_pos_inv_all)

    # Compute the flow vectors to go from the i'th burst image to the base image
    flow_vectors = sample_pos_inv_all - sample_pos_inv_all[:1, ...]

    return burst_images, flow_vectors

def rgb2rawburstdatabase(image, burst_size, downsample_factor=2, burst_transformation_params=None,
                 image_processing_params=None, interpolation_type='bilinear', image_name=None):
    """ Generates a synthetic LR RAW burst from the input image. The input sRGB image is first converted to linear
    sensor space using an inverse camera pipeline. A LR burst is then generated by applying random/specific
    transformations defined by burst_transformation_params to the input image, and downsampling it by the
    downsample_factor. The generated burst is then mosaicekd and corrputed by random noise.

    args:
        image - input sRGB image
        burst_size - Number of images in the output burst
        downsample_factor - Amount of downsampling of the input sRGB image to generate the LR image
        burst_transformation_params - Parameters of the affine transformation used to generate a burst from single image
        image_processing_params - Parameters of the inverse camera pipeline used to obtain RAW image from sRGB image
        interpolation_type - interpolation operator used when performing affine transformations and downsampling
    """

    if image_processing_params is None:
        image_processing_params = {}

    _defaults = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True}
    for k, v in _defaults.items():
        if k not in image_processing_params:
            image_processing_params[k] = v

    # Sample camera pipeline params
    if image_processing_params['random_ccm']:
        if image_processing_params.get('predefined_params', None) is not None:
            rgb2cam = image_processing_params['predefined_params'][image_name]['rgb2cam']
        else:
            rgb2cam = rgb2raw.random_ccm()
    else:
        rgb2cam = torch.eye(3).float()
    cam2rgb = rgb2cam.inverse()

    # Sample gains
    if image_processing_params['random_gains']:
        if image_processing_params.get('predefined_params', None) is not None:
            rgb_gain = image_processing_params['predefined_params'][image_name]['rgb_gain']
            red_gain = image_processing_params['predefined_params'][image_name]['red_gain']
            blue_gain = image_processing_params['predefined_params'][image_name]['blue_gain']
        else:    
            rgb_gain, red_gain, blue_gain = rgb2raw.random_gains()
    else:
        rgb_gain, red_gain, blue_gain = (1.0, 1.0, 1.0)

    # Approximately inverts global tone mapping.
    use_smoothstep = image_processing_params['smoothstep']
    if use_smoothstep:
        image = rgb2raw.invert_smoothstep(image)

    # Inverts gamma compression.
    use_gamma = image_processing_params['gamma']
    if use_gamma:
        image = rgb2raw.gamma_expansion(image)

    # Inverts color correction.
    image = rgb2raw.apply_ccm(image, rgb2cam)

    # Approximately inverts white balance and brightening.
    image = rgb2raw.safe_invert_gains(image, rgb_gain, red_gain, blue_gain)

    # Clip saturated pixels.
    image = image.clamp(0.0, 1.0)

    # Generate LR burst
    # image_burst_rgb, flow_vectors = single2lrburst(image, burst_size=burst_size,
    #                                                downsample_factor=downsample_factor,
    #                                                transformation_params=burst_transformation_params,
    #                                                interpolation_type=interpolation_type)
    image_burst_rgb, flow_vectors = single2lrburstdatabase(image, burst_size=burst_size,
                                                   downsample_factor=downsample_factor,
                                                   transformation_params=burst_transformation_params,
                                                   interpolation_type=interpolation_type)

    # mosaic
    image_burst = rgb2raw.mosaic(image_burst_rgb.clone())

    # Add noise
    if image_processing_params['add_noise']:
        if image_processing_params.get('predefined_params', None) is not None:
            shot_noise_level = image_processing_params['predefined_params'][image_name]['shot_noise_level']
            read_noise_level = image_processing_params['predefined_params'][image_name]['read_noise_level']
        else:
            shot_noise_level, read_noise_level = rgb2raw.random_noise_levels()
        image_burst = rgb2raw.add_noise(image_burst, shot_noise_level, read_noise_level)
    else:
        shot_noise_level = 0
        read_noise_level = 0

    # Clip saturated pixels.
    image_burst = image_burst.clamp(0.0, 1.0)

    meta_info = {'rgb2cam': rgb2cam, 'cam2rgb': cam2rgb, 'rgb_gain': rgb_gain, 'red_gain': red_gain,
                 'blue_gain': blue_gain, 'smoothstep': use_smoothstep, 'gamma': use_gamma,
                 'shot_noise_level': shot_noise_level, 'read_noise_level': read_noise_level}
    return image_burst, image, image_burst_rgb, flow_vectors, meta_info

def flatten_raw_image(im_raw_4ch, return_np=False):
    """ unpack a 4-channel tensor into a single channel bayer image"""
    if isinstance(im_raw_4ch, np.ndarray):
        im_out = np.zeros_like(im_raw_4ch, shape=(im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2))
    elif isinstance(im_raw_4ch, torch.Tensor):
        im_out = torch.zeros((im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2), dtype=im_raw_4ch.dtype)
    else:
        raise Exception

    im_out[0::2, 0::2] = im_raw_4ch[0, :, :]
    im_out[0::2, 1::2] = im_raw_4ch[1, :, :]
    im_out[1::2, 0::2] = im_raw_4ch[2, :, :]
    im_out[1::2, 1::2] = im_raw_4ch[3, :, :]
    if return_np:
        return im_out.numpy()
    else:
        return im_out

def main(ref_img, comp_imgs, options, params):
    """
    This is the implementation of Alg. 1: HandheldBurstSuperResolution.
    Some part of Alg. 2: Registration are also integrated for optimisation.

    Parameters
    ----------
    ref_img : Array[imshape_y, imshape_x]
        Reference frame J_1
    comp_imgs : Array[N-1, imshape_y, imshape_x]
        Remaining frames of the burst J_2, ..., J_N
        
    options : dict
        verbose options.
    params : dict
        paramters.

    Returns
    -------
    num : device Array[imshape_y*s, imshape_y*s, 3]
        generated RGB image WITHOUT any post-processing.
    debug_dict : dict
        Contains (if debugging is enabled) some debugging infos.

    """
    
    grey_method = params['grey method']
    
    ### verbose and timing related stuff
    verbose = options['verbose'] >= 1
    verbose_2 = options['verbose'] >= 2
    verbose_3 = options['verbose'] >= 3
    
    compute_grey_images_ = timer(compute_grey_images, verbose_3, end_s="- Ref grey image estimated by {}".format(grey_method))
    init_block_matching_ = timer(init_block_matching, verbose_2, '\nBeginning Block Matching initialisation', 'Block Matching initialised (Total)')
    init_ICA_ = timer(init_ICA, verbose_2, '\nBeginning ICA initialisation', 'ICA initialised (Total)')
    init_robustness_ = timer(init_robustness, verbose_2, "\nEstimating ref image local stats", 'Local stats estimated (Total)')
    compute_grey_images_ = timer(compute_grey_images, verbose_3, end_s="- grey images estimated by {}".format(grey_method))
    align_image_block_matching_ = timer(align_image_block_matching, verbose_2, 'Beginning block matching', 'Block Matching (Total)')
    ICA_optical_flow_ = timer(ICA_optical_flow, verbose_2, '\nBeginning ICA alignment', 'Image aligned using ICA (Total)')
    compute_robustness_ = timer(compute_robustness, verbose_2, '\nEstimating robustness', 'Robustness estimated (Total)')
    estimate_kernels_ = timer(estimate_kernels, verbose_2, '\nEstimating kernels', 'Kernels estimated (Total)')
    merge_ = timer(merge, verbose_2, '\nAccumulating Image', 'Image accumulated (Total)')
    merge_ref_ = timer(merge_ref, verbose_2, '\nAccumulating ref Img', 'Ref Img accumulated (Total)')    
    divide_ = timer(divide, verbose_2, end_s='\n------------------------\nImage normalized (Total)')
    
    
    bayer_mode = params['mode']=='bayer'
    
    debug_mode = params['debug']
    debug_dict = {"robustness":[],
                  "flow":[]}
    
    accumulate_r = params['accumulated robustness denoiser']['on']

    #### Moving to GPU
    cuda_ref_img = cuda.to_device(ref_img)
    cuda.synchronize()
    
    if verbose :
        print("\nProcessing reference image ---------\n")
        t1 = time.perf_counter()
    
    
    #### Raw to grey
    
    if bayer_mode :
        cuda_ref_grey = compute_grey_images_(cuda_ref_img, grey_method)

    else:
        cuda_ref_grey = cuda_ref_img
        
    #### Block Matching
        
    reference_pyramid = init_block_matching_(cuda_ref_grey, options, params['block matching'])
    
    #### ICA : compute grad and hessian    
    
    ref_gradx, ref_grady, hessian = init_ICA_(cuda_ref_grey, options, params['kanade'])
    
    #### Local stats estimation
    if accumulate_r:
        ref_local_means, ref_local_stds = init_robustness_(cuda_ref_img,options, params['robustness'])
    else:
        _ = init_robustness_(cuda_ref_img,options, params['robustness'])
    
    if accumulate_r:
        accumulated_r = cuda.to_device(np.zeros(ref_local_means.shape[:2]))

    # zeros init of num and den
    scale = params["scale"]
    native_imshape_y, native_imshape_x = cuda_ref_img.shape
    output_size = (round(scale*native_imshape_y), round(scale*native_imshape_x))
    num = cuda.to_device(np.zeros(output_size+(3,), dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    den = cuda.to_device(np.zeros(output_size+(3,), dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    
    if verbose :
        cuda.synchronize()
        getTime(t1, '\nRef Img processed (Total)')
    

    n_images = comp_imgs.shape[0]
    for im_id in range(n_images):
        if verbose :
            cuda.synchronize()
            print("\nProcessing image {} ---------\n".format(im_id+1))
            im_time = time.perf_counter()
        
        #### Moving to GPU
        cuda_img = cuda.to_device(comp_imgs[im_id])
        
        #### Compute Grey Images
        if bayer_mode:
            cuda_im_grey = compute_grey_images(comp_imgs[im_id], grey_method)

        else:
            cuda_im_grey = cuda_img
        
        #### Block Matching
        
        pre_alignment = align_image_block_matching_(cuda_im_grey, reference_pyramid, options, params['block matching'])
        
        #### ICA
        
        cuda_final_alignment = ICA_optical_flow_(cuda_im_grey, cuda_ref_grey,
                                                 ref_gradx, ref_grady,
                                                 hessian, pre_alignment,
                                                 options, params['kanade'])
        
        if debug_mode:
            debug_dict["flow"].append(cuda_final_alignment.copy_to_host())
            
            
        #### Robustness
        cuda_robustness = compute_robustness_(cuda_img, ref_local_means, ref_local_stds, cuda_final_alignment,
                                            options, params['robustness'])
        if accumulate_r:
            add(accumulated_r, cuda_robustness)
        

        #### Kernel estimation
        
        cuda_kernels = estimate_kernels_(cuda_img, options, params['merging'])
        
        #### Merging
        
        merge_(cuda_img, cuda_final_alignment, cuda_kernels, cuda_robustness, num, den,
               options, params['merging'])
        
        if verbose :
            cuda.synchronize()
            getTime(im_time, '\nImage processed (Total)')
            
        if debug_mode : 
            debug_dict['robustness'].append(cuda_robustness.copy_to_host())
    
    #### Ref kernel estimation
        
    cuda_kernels = estimate_kernels_(cuda_ref_img, options, params['merging'])
    
    #### Merge ref

    if accumulate_r:     
        merge_ref_(cuda_ref_img, cuda_kernels,
                   num, den,
                   options, params["merging"], accumulated_r)
    else:
        merge_ref_(cuda_ref_img, cuda_kernels,
                   num, den,
                   options, params["merging"])
    

        
    # num is outwritten into num/den
    divide_(num, den)
    
    if verbose :
        s = '\nTotal ellapsed time : '
        print(s, ' ' * (50 - len(s)), ': ', round((time.perf_counter() - t1), 2), 'seconds')
    
    if accumulate_r :
        debug_dict['accumulated robustness'] = accumulated_r
        
    return num, debug_dict

def calculate_snr(image, shot_noise_level, read_noise_level):
    # Calculate the mean signal level
    signal = np.mean(image)
    print("signal mean: ", signal)
    # Calculate the total noise level
    total_noise = np.sqrt(shot_noise_level**2 + read_noise_level**2)
    print("total_noise: ", total_noise)
    print("shot_noise_level: ", shot_noise_level)
    print("read_noise_level: ", read_noise_level)
    # Calculate the SNR
    snr = signal / total_noise

    return snr

def process(burst_path, options=None, custom_params=None):  
    """
    Processes the burst

    Parameters
    ----------
    burst_path : str or Path
        Path of the folder where the .dng burst is located
    options : dict
        
    params : Parameters
        See params.py for more details.

    Returns
    -------
    Array
        The processed image

    """
    if options is None:
        options = {'verbose' : 0}
    currentTime, verbose_1, verbose_2 = (time.perf_counter(),
                                         options['verbose'] >= 1,
                                         options['verbose'] >= 2)
    params = {}
    
    # reading image stack
    ref_raw, raw_comp, ISO, tags, CFA, xyz2cam, ref_path = load_dng_burst(burst_path)
    print('CFA: ', CFA)
    print("raw ref shape: ", type(ref_raw))  # need to be normalized to 0-1 
    print("raw 1 shape: ", type(raw_comp))
    # if the algorithm had to be run on a specific sensor,
    # the precise values of alpha and beta could be used instead
    if 'mode' in custom_params and custom_params['mode'] == 'grey':
        alpha = tags['Image Tag 0xC761'].values[0][0]
        beta = tags['Image Tag 0xC761'].values[1][0]
    else:
        # Averaging RGB noise values
        ## IMPORTANT NOTE : the noise model exif already are NOT for nominal ISO 100
        ## But are already scaled for the image ISO.
        alpha = sum([x[0] for x in tags['Image Tag 0xC761'].values[::2]])/3
        beta = sum([x[0] for x in tags['Image Tag 0xC761'].values[1::2]])/3
    print("alpha: ", alpha)
    print("beta: ", beta)
    #### Packing noise model related to picture ISO
    # curve_iso = round_iso(ISO) # Rounds non standart ISO to regular ISO (100, 200, 400, ...)
    # std_noise_model_label = 'noise_model_std_ISO_{}'.format(curve_iso)
    # diff_noise_model_label = 'noise_model_diff_ISO_{}'.format(curve_iso)
    # std_noise_model_path = (NOISE_MODEL_PATH / std_noise_model_label).with_suffix('.npy')
    # diff_noise_model_path = (NOISE_MODEL_PATH / diff_noise_model_label).with_suffix('.npy')
    
    # std_curve = np.load(std_noise_model_path)
    # diff_curve = np.load(diff_noise_model_path)
    
    # Use this to compute noise curves on the fly
    std_curve, diff_curve = run_fast_MC(alpha, beta)
    
    
    if verbose_2:
        currentTime = getTime(currentTime, ' -- Read raw files')



    
    #### Estimating ref image SNR
    brightness = np.mean(ref_raw)
    
    id_noise = round(1000*brightness)
    std = std_curve[id_noise]
    
    SNR = brightness/std
    if verbose_1:
        print(" ",10*"-")
        print('|ISO : {}'.format(ISO))
        print('|Image brightness : {:.2f}'.format(brightness))
        print('|expected noise std : {:.2e}'.format(std))
        print('|Estimated SNR : {:.2f}'.format(SNR))
    
    SNR_params = get_params(SNR)
    
    #### Merging params dictionnaries
    
    # checking (just in case !)
    check_params_validity(SNR_params, ref_raw.shape)
    
    if custom_params is not None :
        params = merge_params(dominant=custom_params, recessive=SNR_params)
        check_params_validity(params, ref_raw.shape)
        
    #### adding metadatas to dict 
    if not 'noise' in params['merging'].keys(): 
        params['merging']['noise'] = {}

        
    params['merging']['noise']['alpha'] = alpha
    params['merging']['noise']['beta'] = beta
    
    ## Writing exifs data into parameters
    if not 'exif' in params['merging'].keys(): 
        params['merging']['exif'] = {}
    if not 'exif' in params['robustness'].keys(): 
        params['robustness']['exif'] = {}
        
    params['merging']['exif']['CFA Pattern'] = CFA
    params['robustness']['exif']['CFA Pattern'] = CFA
    params['ISO'] = ISO
    
    params['robustness']['std_curve'] = std_curve
    params['robustness']['diff_curve'] = diff_curve
    
    # copying parameters values in sub-dictionaries
    if 'scale' not in params["merging"].keys() :
        params["merging"]["scale"] = params["scale"]
    if 'scale' not in params['accumulated robustness denoiser'].keys() :
        params['accumulated robustness denoiser']["scale"] = params["scale"]
    if 'tileSize' not in params["kanade"]["tuning"].keys():
        params["kanade"]["tuning"]['tileSize'] = params['block matching']['tuning']['tileSizes'][0]
    if 'tileSize' not in params["robustness"]["tuning"].keys():
        params["robustness"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']
    if 'tileSize' not in params["merging"]["tuning"].keys():
        params["merging"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']


    if 'mode' not in params["kanade"].keys():
        params["kanade"]["mode"] = params['mode']
    if 'mode' not in params["robustness"].keys():
        params["robustness"]["mode"] = params['mode']
    if 'mode' not in params["merging"].keys():
        params["merging"]["mode"] = params['mode']
    if 'mode' not in params['accumulated robustness denoiser'].keys():
        params['accumulated robustness denoiser']["mode"] = params['mode']
    
    # deactivating robustness accumulation if robustness is disabled
    params['accumulated robustness denoiser']['median']['on'] &= params['robustness']['on']
    params['accumulated robustness denoiser']['gauss']['on'] &= params['robustness']['on']
    params['accumulated robustness denoiser']['merge']['on'] &= params['robustness']['on']
    
    params['accumulated robustness denoiser']['on'] = \
        (params['accumulated robustness denoiser']['gauss']['on'] or
         params['accumulated robustness denoiser']['median']['on'] or
         params['accumulated robustness denoiser']['merge']['on'])
     
    # if robustness aware denoiser is in merge mode, copy in merge params
    if params['accumulated robustness denoiser']['merge']['on']:
        params['merging']['accumulated robustness denoiser'] = params['accumulated robustness denoiser']['merge']
    else:
        params['merging']['accumulated robustness denoiser'] = {'on' : False}
        
        
        
    
    
    #### Running the handheld pipeline
    handheld_output, debug_dict = main(ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE), raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE), options, params)
    
    print("handheld output shape: ", handheld_output.shape)
    #### Performing frame count aware denoising if enabled
    median_params = params['accumulated robustness denoiser']['median']
    gauss_params = params['accumulated robustness denoiser']['gauss']
    
    median = median_params['on']
    gauss = gauss_params['on']
    post_frame_count_denoise = (median or gauss)
    
    params_pp = params['post processing']
    post_processing_enabled = params_pp['on']
    
    if post_frame_count_denoise or post_processing_enabled:
        if verbose_1:
            print('Beginning post processing')
    
    if post_frame_count_denoise : 
        if verbose_2:
            print('-- Robustness aware bluring')
        
        if median:
            handheld_output = frame_count_denoising_median(handheld_output, debug_dict['accumulated robustness'],
                                                           median_params)
        if gauss:
            handheld_output = frame_count_denoising_gauss(handheld_output, debug_dict['accumulated robustness'],
                                                          gauss_params)


    #### post processing
    
    if post_processing_enabled:
        if verbose_2:
            print('-- Post processing image')
        
        raw = rawpy.imread(ref_path)
        output_image = raw2rgb.postprocess(raw, handheld_output.copy_to_host(),
                                           params_pp['do color correction'],
                                           params_pp['do tonemapping'],
                                           params_pp['do gamma'],
                                           params_pp['do sharpening'],
                                           params_pp['do devignette'],
                                           xyz2cam,
                                           params_pp['sharpening']
                                           ) 
    else:
        output_image = handheld_output.copy_to_host()
        
    # # Applying image orientation
    # if 'Image Orientation' in tags.keys():
    #     ori = tags['Image Orientation'].values[0]
    #     print("Image Orientation: ", ori)
    # else:
    #     ori = 1
    #     warnings.warns('The Image Orientation EXIF tag could not be found. \
    #                   The image may be mirrored or misoriented.')
    # output_image = apply_orientation(output_image, ori)
    # if 'accumulated robustness' in debug_dict.keys():
    #     debug_dict['accumulated robustness'] = apply_orientation(debug_dict['accumulated robustness'], ori)
    
    
    
    # if params['debug']:
    #     return output_image, debug_dict
    # else:
    #     return output_image

def processRGB(rgb_img, image_name=None, burst_transformation_params=None, image_processing_params=None, burst_size=4, downsample_factor=2, crop_sz=None, options=None, custom_params=None):
    """
    Processes the burst

    Parameters
    ----------
    rgb_img : np array of a RGB image (H, W, 3)
    options : dict
        
    params : Parameters
        See params.py for more details.

    Returns
    -------
    Array
        The processed image

    """
    if options is None:
        options = {'verbose' : 0}
    currentTime, verbose_1, verbose_2 = (time.perf_counter(),
                                         options['verbose'] >= 1,
                                         options['verbose'] >= 2)
    params = {}
    
    assert crop_sz is not None, "You must specify the crop size"
    crop_sz = [c + 2 * burst_transformation_params.get('border_crop', 0) for c in crop_sz]
    print("original image size: ", rgb_img.shape)
    if isinstance(rgb_img, np.ndarray):
        rgb_img = torch.from_numpy(rgb_img.transpose((2, 0, 1)))
    
    frame_crop = center_crop(rgb_img, crop_sz)
    frame_crop = frame_crop / 255.0

    # generate burst images
    burst, frame_gt, burst_rgb, flow_vector, meta_info = rgb2rawburstdatabase(frame_crop,
                                                                            burst_size,
                                                                            downsample_factor,
                                                                            burst_transformation_params=burst_transformation_params,
                                                                            image_processing_params=image_processing_params,
                                                                            interpolation_type='bilinear',
                                                                            image_name=image_name
                                                                            )
    
    # *both ref_raw and burst are in (0,1)
    xyz2cam = meta_info['rgb2cam']
    CFA = np.array([[0,1],
                    [1,2]])
    raw_comp = []
    for idx, img in enumerate(burst):
        if idx == 0:
            ref_raw = flatten_raw_image(img, return_np=True)
        else:
            tmp = flatten_raw_image(img, return_np=True)
            raw_comp.append(tmp)
    raw_comp = np.array(raw_comp)
    # # reading image stack
    # ref_raw, raw_comp, _, _, CFA, _, _ = load_dng_burst(burst_path)
    

    
    alpha = meta_info['shot_noise_level']
    beta = meta_info['read_noise_level']
    SNR = calculate_snr(ref_raw, meta_info['shot_noise_level'], meta_info['read_noise_level'])
    SNR_params = get_params(SNR)
    print("ori SNR: ", SNR)
    assert SNR > 22, "cannot perform SNR less than 22"
    std_curve, diff_curve = run_fast_MC(alpha, beta)
    
    #### Estimating ref image SNR
    brightness = np.mean(ref_raw)
    
    id_noise = round(1000*brightness)
    std = std_curve[id_noise]
    
    SNR = brightness/std
    print("new SNR: ", SNR)

    
    #### Merging params dictionnaries
    
    
    if custom_params is not None :
        params = merge_params(dominant=custom_params, recessive=SNR_params)
        check_params_validity(params, ref_raw.shape)
        
    #### adding metadatas to dict 
    if not 'noise' in params['merging'].keys(): 
        params['merging']['noise'] = {}

        
    params['merging']['noise']['alpha'] = alpha
    params['merging']['noise']['beta'] = beta
    
    ## Writing exifs data into parameters
    if not 'exif' in params['merging'].keys(): 
        params['merging']['exif'] = {}
    if not 'exif' in params['robustness'].keys(): 
        params['robustness']['exif'] = {}
        
    params['merging']['exif']['CFA Pattern'] = CFA
    params['robustness']['exif']['CFA Pattern'] = CFA
    # params['ISO'] = ISO
    
    params['robustness']['std_curve'] = std_curve
    params['robustness']['diff_curve'] = diff_curve
    
    # copying parameters values in sub-dictionaries
    if 'scale' not in params["merging"].keys() :
        params["merging"]["scale"] = params["scale"]
    if 'scale' not in params['accumulated robustness denoiser'].keys() :
        params['accumulated robustness denoiser']["scale"] = params["scale"]
    if 'tileSize' not in params["kanade"]["tuning"].keys():
        params["kanade"]["tuning"]['tileSize'] = params['block matching']['tuning']['tileSizes'][0]
    if 'tileSize' not in params["robustness"]["tuning"].keys():
        params["robustness"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']
    if 'tileSize' not in params["merging"]["tuning"].keys():
        params["merging"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']


    if 'mode' not in params["kanade"].keys():
        params["kanade"]["mode"] = params['mode']
    if 'mode' not in params["robustness"].keys():
        params["robustness"]["mode"] = params['mode']
    if 'mode' not in params["merging"].keys():
        params["merging"]["mode"] = params['mode']
    if 'mode' not in params['accumulated robustness denoiser'].keys():
        params['accumulated robustness denoiser']["mode"] = params['mode']
    
    # deactivating robustness accumulation if robustness is disabled
    params['accumulated robustness denoiser']['median']['on'] &= params['robustness']['on']
    params['accumulated robustness denoiser']['gauss']['on'] &= params['robustness']['on']
    params['accumulated robustness denoiser']['merge']['on'] &= params['robustness']['on']
    
    params['accumulated robustness denoiser']['on'] = \
        (params['accumulated robustness denoiser']['gauss']['on'] or
         params['accumulated robustness denoiser']['median']['on'] or
         params['accumulated robustness denoiser']['merge']['on'])
     
    # if robustness aware denoiser is in merge mode, copy in merge params
    if params['accumulated robustness denoiser']['merge']['on']:
        params['merging']['accumulated robustness denoiser'] = params['accumulated robustness denoiser']['merge']
    else:
        params['merging']['accumulated robustness denoiser'] = {'on' : False}
        
        
        
    
    
    #### Running the handheld pipeline
    handheld_output, debug_dict = main(ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE), raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE), options, params)
    
    print("handheld output shape: ", handheld_output.shape)
    #### Performing frame count aware denoising if enabled
    median_params = params['accumulated robustness denoiser']['median']
    gauss_params = params['accumulated robustness denoiser']['gauss']
    
    median = median_params['on']
    gauss = gauss_params['on']
    post_frame_count_denoise = (median or gauss)
    
    params_pp = params['post processing']
    post_processing_enabled = params_pp['on']
    
    if post_frame_count_denoise or post_processing_enabled:
        if verbose_1:
            print('Beginning post processing')
    
    if post_frame_count_denoise : 
        if verbose_2:
            print('-- Robustness aware bluring')
        
        if median:
            handheld_output = frame_count_denoising_median(handheld_output, debug_dict['accumulated robustness'],
                                                           median_params)
        if gauss:
            handheld_output = frame_count_denoising_gauss(handheld_output, debug_dict['accumulated robustness'],
                                                          gauss_params)


    #### post processing
    
    if post_processing_enabled:
        if verbose_2:
            print('-- Post processing image')
        
        raw = rawpy.imread(ref_path)
        output_image = raw2rgb.postprocess(raw, handheld_output.copy_to_host(),
                                           params_pp['do color correction'],
                                           params_pp['do tonemapping'],
                                           params_pp['do gamma'],
                                           params_pp['do sharpening'],
                                           params_pp['do devignette'],
                                           xyz2cam,
                                           params_pp['sharpening']
                                           ) 
    else:
        output_image = handheld_output.copy_to_host()
        
    # Applying image orientation
    # if 'Image Orientation' in tags.keys():
    #     ori = tags['Image Orientation'].values[0]
    # else:
    #     ori = 1
    #     warnings.warns('The Image Orientation EXIF tag could not be found. \
    #                   The image may be mirrored or misoriented.')
    # output_image = apply_orientation(output_image, ori)
    # if 'accumulated robustness' in debug_dict.keys():
    #     debug_dict['accumulated robustness'] = apply_orientation(debug_dict['accumulated robustness'], ori)
    
    
    
    if params['debug']:
        return output_image, debug_dict
    else:
        return output_image, frame_gt
    
    
