# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:05:08 2020

@author: hp
"""


import openslide
import pandas as pd
import numpy as np
import os 
from PIL import Image, ImageDraw, ImageFont
import PIL
import re
import math

import random

# =============================================================================
#                      数据准备时需要的  funciotn   start
# =============================================================================
def get_filelist(path,regex_ = r"mrxs"):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files: 
            
            Filelist.append(os.path.join(home, filename))
             
    temp_list = np.array(Filelist)
    index_need = np.where(list(map(lambda x: re.search(regex_, x)!=None,temp_list)))
    temp_list = temp_list[index_need[0]]
    temp_list = pd.DataFrame(temp_list)
    # temp_list.columns = ['file_addr']
    return temp_list

def unique_sample(x):
    temp0 =  os.path.split(x) 
    temp1 =  os.path.split(temp0[0])
    temp2 =  temp0[1].split('.')
    return "_".join([temp1[1],temp2[0]])
 
def create_path(base_dir,file_name):
    train_dir = os.path.join(base_dir, file_name)
    train_dir1 = os.path.join(train_dir, 'Case')
    train_dir2 = os.path.join(train_dir, 'Control')
    if os.path.exists(train_dir1)==False:
        os.makedirs(train_dir1 )
    if os.path.exists(train_dir2)==False:
        os.makedirs(train_dir2)
    return (train_dir1,train_dir2)


# #  scale raw slide to small png
# def slide_to_scaled_pil_image(slide_adr,SCALE_FACTOR=20):
#     slide_df = openslide.open_slide(slide_adr)
#     large_w, large_h = slide_df.dimensions
#     new_w = math.floor(large_w / SCALE_FACTOR)
#     new_h = math.floor(large_h / SCALE_FACTOR)
#     level = slide_df.get_best_level_for_downsample(SCALE_FACTOR)
#     whole_slide_image = slide_df.read_region((0, 0), level, slide_df.level_dimensions[level])
#     whole_slide_image = whole_slide_image.convert("RGB")
#     img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
#     return img, large_w, large_h, new_w, new_h ,slide_df
   
# # Raw_file = r'G:\MPN\MPN_Phase1_raw_data\01\1M01.mrxs'
# # out_scale = r'G:\MPN\Phase1_module_2\Data\filter_tile_summary' 
# # scale_factor = 8
# # tile_size = 512
# # Sample_id = '01_1M01'
# # out_tiles = r'G:\MPN\Phase1_module_2\Data\Data_temp\Train\Case'
# def Scale_first_fliter(Raw_file,out_scale,out_tiles,
#                        Sample_id,
#                        scale_factor,tile_size): 
#     img_scaled, raw_w, raw_h, new_w, new_h ,slide_raw= slide_to_scaled_pil_image(Raw_file,scale_factor)
#     addr_scaled = os.path.join(out_scale, Sample_id + '.png')
#     img_scaled.save(addr_scaled) 
#     # only get part 2
#     W_list = list(range(0,raw_w,tile_size))
#     H_list = list(range(int(raw_h/3),int(2*raw_h/3),tile_size)) 
#     tile_numb = 0
#     for W_list_i in W_list:
#         for H_list_i in H_list: 
#             temp_tile = slide_raw.read_region((W_list_i, H_list_i), 0, 
#                                               (tile_size, tile_size))
            
#             num_temp_tile = np.array(temp_tile)
#             if np.max(num_temp_tile) !=0 and np.min(num_temp_tile) !=240:
                
#                 pil_img = temp_tile.convert("RGB")
#                 File_tile_i = Sample_id + '_num_'+str(tile_numb) + '.png'
#                 file_name_tile = os.path.join(out_tiles,File_tile_i)
#                 pil_img.save(file_name_tile)
#                 tile_numb = tile_numb+1
 
                
#     slide_raw.close()
#     return addr_scaled
    
# =============================================================================
#                      数据准备时需要的  funciotn    end
# =============================================================================


import scipy
import openslide  
import numpy 
import os
import re
import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import PIL
import math
import datetime
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
# import scipy.signal
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation
from enum import Enum
# import multiprocessing

from scipy import ndimage as ndi    
from skimage.segmentation import slic

# import scipy.stats as st

Image.MAX_IMAGE_PIXELS = None

# import sklearn
# print(scipy.__version__) 安装1.5.0


def get_filelist(path,regex_ = r"mrxs"):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files: 
            
            Filelist.append(os.path.join(home, filename))
            
            
    temp_list = np.array(Filelist)
    index_need = np.where(list(map(lambda x: re.search(regex_, x)!=None,temp_list)))
    temp_list = temp_list[index_need[0]]
    temp_list = pd.DataFrame(temp_list)
    # temp_list.columns = ['file_addr']
    return temp_list

 

class Time:
    """
    Class for displaying elapsed time.
    """

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed



 
## prepare data 
# Raw_slide_addr = r'E:\MPN\Raw_data\ywei20200916'
# slide_suffix = r"mrxs"
# 获得slide path list
def get_slide_path(Raw_slide_addr,slide_suffix=None): 
    temp_list = np.array(os.listdir(Raw_slide_addr))
    if slide_suffix != None: 
        index_need = np.where(list(map(lambda x: re.search(slide_suffix, x)!=None,temp_list)))
        temp_list = temp_list[index_need[0]]
        slide_filepath = list(map(lambda x: os.path.join(Raw_slide_addr, x) ,
                                  temp_list))
        return slide_filepath
    
    slide_filepath = list(map(lambda x: os.path.join(Raw_slide_addr, x) ,
                                  temp_list))
    
    return slide_filepath

#  scale raw slide to small png
def slide_to_scaled_pil_image(slide_adr,SCALE_FACTOR=20):
    slide_df = openslide.open_slide(slide_adr)
    large_w, large_h = slide_df.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide_df.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = slide_df.read_region((0, 0), level, slide_df.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, large_w, large_h, new_w, new_h

def np_to_pil(np_img,ADDITIONAL_NP_STATS=False):
    """
    Convert a NumPy array to a PIL Image.

    Args:
        np_img: The image represented as a NumPy array.

    Returns:
         The NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)


def pil_to_np_rgb(pil_img):
    """
    Convert a PIL Image to a NumPy array.

    Note that RGB PIL (w, h) -> NumPy (h, w, 3).

    Args:
        pil_img: The PIL Image.

    Returns:
        The PIL image converted to a NumPy array.
    """
    t = Time()
    rgb = np.asarray(pil_img)
    np_info(rgb, "RGB", t.elapsed())
    return rgb








def np_info(np_arr, name=None, elapsed=None,ADDITIONAL_NP_STATS=False):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.

    Args:
        np_arr: The NumPy array.
        name: The (optional) name of the array.
        elapsed: The (optional) time elapsed to perform a filtering operation.
    """

    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"

    if ADDITIONAL_NP_STATS is False:
        print("%-20s | Time: %-14s    Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
    else:
        # np_arr = np.asarray(np_arr)
        max = np_arr.max()
        min = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print("%-20s | Time: %-14s Min: %6.2f    Max: %6.2f    Mean: %6.2f    Binary: %s    Type: %-7s Shape: %s" % (
            name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))






# filename = 'E:\\MPN\\Raw_data\\ywei20200916\\1M02.mrxs'

def open_image_np(filename):
    """
    Open an image (*.jpg, *.png, etc) as an RGB NumPy array.

    Args:
        filename: Name of the image file.

    returns:
        A NumPy representing an RGB image.
    """
    pil_img = Image.open(filename)
    np_img = pil_to_np_rgb(pil_img)
    return(np_img)


 

def display_img(np_img, text=None, 
                                font_path="C:/Windows/Fonts/Arial.ttf", 
                                size=48, color=(255, 0, 0),
                                background=(255, 255, 255), 
                                border=(0, 0, 0), bg=False):
    """
    Convert a NumPy array to a PIL image, add text to the image, and display the image.

    Args:
        np_img: Image as a NumPy array.
        text: The text to add to the image.
        font_path: The path to the font to use.
        size: The font size
        color: The font color
        background: The background color
        border: The border color
        bg: If True, add rectangle background behind text
    """
    result = np_to_pil(np_img)
    # if gray, convert to RGB for display
    if result.mode == 'L':
        result = result.convert('RGB')
    draw = ImageDraw.Draw(result)
    if text is not None:
        font = ImageFont.truetype(font_path, size)
        if bg:
            (x, y) = draw.textsize(text, font)
            draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
        draw.text((2, 0), text, color, font=font)
    result.show()


# Filters 
def filter_rgb_to_grayscale(np_img, output_type="uint8"):
    """
    Convert an RGB NumPy array to a grayscale NumPy array.

    Shape (h, w, c) to (h, w).

    Args:
        np_img: RGB Image as a NumPy array.
        output_type: Type of array to return (float or uint8)

    Returns:
        Grayscale image as NumPy array with shape (h, w).
    """
    t = Time()
    # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
    grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
    if output_type != "float":
        grayscale = grayscale.astype("uint8")
    np_info(grayscale, "Gray", t.elapsed())
    return grayscale


def filter_complement(np_img, output_type="uint8"):
    """
    Obtain the complement of an image as a NumPy array.

    Args:
        np_img: Image as a NumPy array.
        type: Type of array to return (float or uint8).

    Returns:
        Complement image as Numpy array.
    """
    t = Time()
    if output_type == "float":
        complement = 1.0 - np_img
    else:
        complement = 255 - np_img
    np_info(complement, "Complement", t.elapsed())
    return complement


def filter_threshold(np_img, threshold, output_type="bool"):
    """
    Return mask where a pixel has a value if it exceeds the threshold value.

    Args:
        np_img: Binary image as a NumPy array.
        threshold: The threshold value to exceed.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
        pixel exceeds the threshold value.
    """
    t = Time()
    result = (np_img > threshold)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Threshold", t.elapsed())
    return result


def filter_hysteresis_threshold(np_img, low=50, high=100, output_type="uint8"):
    """
    Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.

    Args:
        np_img: Image as a NumPy array.
        low: Low threshold.
        high: High threshold.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
    """
    t = Time()
    hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
    if output_type == "bool":
        pass
    elif output_type == "float":
        hyst = hyst.astype(float)
    else:
        hyst = (255 * hyst).astype("uint8")
    np_info(hyst, "Hysteresis Threshold", t.elapsed())
    return hyst




def filter_otsu_threshold(np_img, output_type="uint8"):
    """
    Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.

    Args:
        np_img: Image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
    """
    t = Time()
    otsu_thresh_value = sk_filters.threshold_otsu(np_img)
    otsu = (np_img > otsu_thresh_value)
    if output_type == "bool":
        pass
    elif output_type == "float":
        otsu = otsu.astype(float)
    else:
        otsu = otsu.astype("uint8") * 255
    np_info(otsu, "Otsu Threshold", t.elapsed())
    return otsu


def filter_local_otsu_threshold(np_img, disk_size=3, output_type="uint8"):
    """
    Compute local Otsu threshold for each pixel and return binary image based on pixels being less than the
    local Otsu threshold.

    Args:
        np_img: Image as a NumPy array.
        disk_size: Radius of the disk structuring element used to compute the Otsu threshold for each pixel.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where local Otsu threshold values have been applied to original image.
    """
    t = Time()
    local_otsu = sk_filters.rank.otsu(np_img, sk_morphology.disk(disk_size))
    if output_type == "bool":
        pass
    elif output_type == "float":
        local_otsu = local_otsu.astype(float)
    else:
        local_otsu = local_otsu.astype("uint8") * 255
    np_info(local_otsu, "Otsu Local Threshold", t.elapsed())
    return local_otsu


def filter_entropy(np_img, neighborhood=9, threshold=5, output_type="uint8"):
    """
    Filter image based on entropy (complexity).

    Args:
        np_img: Image as a NumPy array.
        neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
        threshold: Threshold value.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
    """
    t = Time()
    entr = sk_filters.rank.entropy(np_img, np.ones((neighborhood, neighborhood))) > threshold
    if output_type == "bool":
        pass
    elif output_type == "float":
        entr = entr.astype(float)
    else:
        entr = entr.astype("uint8") * 255
    np_info(entr, "Entropy", t.elapsed())
    return entr


def filter_canny(np_img, sigma=1, low_threshold=0, high_threshold=25, output_type="uint8"):
    """
    Filter image based on Canny algorithm edges.

    Args:
        np_img: Image as a NumPy array.
        sigma: Width (std dev) of Gaussian.
        low_threshold: Low hysteresis threshold value.
        high_threshold: High hysteresis threshold value.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
    """
    t = Time()
    can = sk_feature.canny(np_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    if output_type == "bool":
        pass
    elif output_type == "float":
        can = can.astype(float)
    else:
        can = can.astype("uint8") * 255
    np_info(can, "Canny Edges", t.elapsed())
    return can




# scale
# raw_slide_path = r'E:\\MPN\\Raw_data\\ywei20200916\\1M02.mrxs'
def combine_save_path_file(raw_slide_path,
                           unique_sample_id,
                           base_addr,
                           large_w, 
                           large_h,
                           new_w, 
                           new_h,order_,
                           SCALE_FACTOR=32,
                           Save_Dir = "scale",
                           type_ = 'png'):
    assert Save_Dir in ["scale","tiles","filter","summary"],"Save_Dir must one of 'scale' 'tiles' 'filter' 'summary' "
    
    # _,File_name = os.path.split(raw_slide_path)
    # File_name.split(".")
    
    img_path = os.path.join(base_addr, Save_Dir ,
                            unique_sample_id +"-r-" + str(SCALE_FACTOR) + 
                            str(large_w) + "-r-" + str(large_h) + "-n-" + 
                            str(new_w) + "-n-" + str(new_h) +'-'+str(order_) +"." + type_)
    return(img_path)


# file_name_list = temp_slide
# slide_range_to_images(file_name_list, 20,base_addr)
def slide_range_to_images(file_name_list,SCALE_FACTOR,base_addr,sample_id_list):
    """
    Convert a range of  slides to smaller images (in a format such as jpg or png).

    Args:
            start_ind: Starting index (inclusive).
            end_ind: Ending index (inclusive).

    Returns:
            The starting index and the ending index of the slides that were converted.
    """
    # Count_ =1
    # for slide_num in file_name_list:
    #     Scale_slide_to_image(slide_num,SCALE_FACTOR,base_addr,order_ = Count_)
    #     Count_ = Count_+1 
    for i in range(len(sample_id_list)):
        Scale_slide_to_image(file_name_list[i],
                             SCALE_FACTOR,
                             base_addr,
                             order_ = i,
                             unique_sample_id = sample_id_list[i])
    # return(file_name_list)

# slide_addr = All_Raw_file[1]
# base_addr = r'E:\MPN\Raw_data\test_own_code'
# SCALE_FACTOR = 32
def Scale_slide_to_image(slide_addr,SCALE_FACTOR,base_addr,order_,unique_sample_id):
    """
    Convert a slide to a saved scaled-down image in a format such as jpg or png.
    Args:
        slide_addr: The slide number.
    """
    img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_addr,SCALE_FACTOR=SCALE_FACTOR)
    scale_path = os.path.join(base_addr, "scale")
    if not os.path.exists(scale_path):
        os.makedirs(scale_path)
    img_path_scale = combine_save_path_file(raw_slide_path=slide_addr, 
                                      base_addr = base_addr,
                                      large_w=large_w, 
                                      large_h=large_h, 
                                      new_w=new_w, 
                                      new_h=new_h,
                                      SCALE_FACTOR=SCALE_FACTOR,
                                      Save_Dir = "scale",order_=order_,
                                      unique_sample_id = unique_sample_id)

    
    print("Saving image to: " + img_path_scale) 
    img.save(img_path_scale)
   
    



# fliter

def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

    Args:
        np_img: Image as a NumPy array.

    Returns:
        The percentage of the NumPy array that is masked.
    """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
    """
    Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
    and eosin are purplish and pinkish, which do not have much green to them.

    Args:
        np_img: RGB image as a NumPy array.
        green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
        avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
        overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """
    t = Time()

    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        print(
            "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
                mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
        gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    np_info(np_img, "Filter Green Channel", t.elapsed())
    return np_img




def mask_rgb(rgb, mask):
   
    t = Time()
    result = rgb * np.dstack([mask, mask, mask])
    np_info(result, "Mask RGB", t.elapsed())
    return result







def filter_hed_to_hematoxylin(np_img, output_type="uint8"):
    """
    Obtain Hematoxylin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
    contrast.

    Args:
        np_img: HED image as a NumPy array.
        output_type: Type of array to return (float or uint8).

    Returns:
        NumPy array for Hematoxylin channel.
    """
    t = Time()
    hema = np_img[:, :, 0]
    if output_type == "float":
        hema = sk_exposure.rescale_intensity(hema, out_range=(0.0, 1.0))
    else:
        hema = (sk_exposure.rescale_intensity(hema, out_range=(0, 255))).astype("uint8")
    np_info(hema, "HED to Hematoxylin", t.elapsed())
    return hema


def filter_hed_to_eosin(np_img, output_type="uint8"):
    """
    Obtain Eosin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
    contrast.

    Args:
        np_img: HED image as a NumPy array.
        output_type: Type of array to return (float or uint8).

    Returns:
        NumPy array for Eosin channel.
    """
    t = Time()
    eosin = np_img[:, :, 1]
    if output_type == "float":
        eosin = sk_exposure.rescale_intensity(eosin, out_range=(0.0, 1.0))
    else:
        eosin = (sk_exposure.rescale_intensity(eosin, out_range=(0, 255))).astype("uint8")
    np_info(eosin, "HED to Eosin", t.elapsed())
    return eosin


def filter_binary_fill_holes(np_img, output_type="bool"):
    """
    Fill holes in a binary object (bool, float, or uint8).

    Args:
        np_img: Binary image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where holes have been filled.
    """
    t = Time()
    if np_img.dtype == "uint8":
        np_img = np_img / 255
    result = sc_morph.binary_fill_holes(np_img)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Binary Fill Holes", t.elapsed())
    return result


def filter_binary_erosion(np_img, disk_size=5, iterations=1, output_type="uint8"):
    """
    Erode a binary object (bool, float, or uint8).

    Args:
        np_img: Binary image as a NumPy array.
        disk_size: Radius of the disk structuring element used for erosion.
        iterations: How many times to repeat the erosion.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where edges have been eroded.
    """
    t = Time()
    if np_img.dtype == "uint8":
        np_img = np_img / 255
    result = sc_morph.binary_erosion(np_img, sk_morphology.disk(disk_size), iterations=iterations)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Binary Erosion", t.elapsed())
    return result


def filter_binary_dilation(np_img, disk_size=5, iterations=1, output_type="uint8"):
    """
    Dilate a binary object (bool, float, or uint8).

    Args:
        np_img: Binary image as a NumPy array.
        disk_size: Radius of the disk structuring element used for dilation.
        iterations: How many times to repeat the dilation.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where edges have been dilated.
    """
    t = Time()
    if np_img.dtype == "uint8":
        np_img = np_img / 255
    result = sc_morph.binary_dilation(np_img, sk_morphology.disk(disk_size), iterations=iterations)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Binary Dilation", t.elapsed())
    return result


def filter_binary_opening(np_img, disk_size=3, iterations=1, output_type="uint8"):
    """
    Open a binary object (bool, float, or uint8). Opening is an erosion followed by a dilation.
    Opening can be used to remove small objects.

    Args:
        np_img: Binary image as a NumPy array.
        disk_size: Radius of the disk structuring element used for opening.
        iterations: How many times to repeat.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) following binary opening.
    """
    t = Time()
    if np_img.dtype == "uint8":
        np_img = np_img / 255
    result = sc_morph.binary_opening(np_img, sk_morphology.disk(disk_size), iterations=iterations)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Binary Opening", t.elapsed())
    return result


def filter_binary_closing(np_img, disk_size=3, iterations=1, output_type="uint8"):
    """
    Close a binary object (bool, float, or uint8). Closing is a dilation followed by an erosion.
    Closing can be used to remove small holes.

    Args:
        np_img: Binary image as a NumPy array.
        disk_size: Radius of the disk structuring element used for closing.
        iterations: How many times to repeat.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) following binary closing.
    """
    t = Time()
    if np_img.dtype == "uint8":
        np_img = np_img / 255
    result = sc_morph.binary_closing(np_img, sk_morphology.disk(disk_size), iterations=iterations)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Binary Closing", t.elapsed())
    return result


def filter_kmeans_segmentation(np_img, compactness=10, n_segments=800):
    """
    Use K-means segmentation (color/space proximity) to segment RGB image where each segment is
    colored based on the average color for that segment.

    Args:
        np_img: Binary image as a NumPy array.
        compactness: Color proximity versus space proximity factor.
        n_segments: The number of segments.

    Returns:
        NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
        color for that segment.
    """
    t = Time()
    labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
    result = sk_color.label2rgb(labels, np_img, kind='avg')
    np_info(result, "K-Means Segmentation", t.elapsed())
    return result


def filter_rag_threshold(np_img, compactness=10, n_segments=800, threshold=9):
    """
    Use K-means segmentation to segment RGB image, build region adjacency graph based on the segments, combine
    similar regions based on threshold value, and then output these resulting region segments.

    Args:
        np_img: Binary image as a NumPy array.
        compactness: Color proximity versus space proximity factor.
        n_segments: The number of segments.
        threshold: Threshold value for combining regions.

    Returns:
        NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
        color for that segment (and similar segments have been combined).
    """
    t = Time()
    labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
    g = sk_future.graph.rag_mean_color(np_img, labels)
    labels2 = sk_future.graph.cut_threshold(labels, g, threshold)
    result = sk_color.label2rgb(labels2, np_img, kind='avg')
    np_info(result, "RAG Threshold", t.elapsed())
    return result


def filter_threshold(np_img, threshold, output_type="bool"):
    """
    Return mask where a pixel has a value if it exceeds the threshold value.

    Args:
        np_img: Binary image as a NumPy array.
        threshold: The threshold value to exceed.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
        pixel exceeds the threshold value.
    """
    t = Time()
    result = (np_img > threshold)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Threshold", t.elapsed())
    return result


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
    """
    Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
    and eosin are purplish and pinkish, which do not have much green to them.

    Args:
        np_img: RGB image as a NumPy array.
        green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
        avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
        overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
    """
    t = Time()

    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        print(
            "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
                mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
        gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    np_info(np_img, "Filter Green Channel", t.elapsed())
    return np_img


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
                             display_np_info=False):
    """
    Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
    red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_lower_thresh: Red channel lower threshold value.
        green_upper_thresh: Green channel upper threshold value.
        blue_upper_thresh: Blue channel upper threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] > red_lower_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Red", t.elapsed())
    return result


def filter_red_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out red pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    t = Time()
    result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
                     filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
                     filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
                     filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
                     filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
                     filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
                     filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
                     filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
                     filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Red Pen", t.elapsed())
    return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                                 display_np_info=False):
    """
    Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
    red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
    Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
    lower threshold value rather than a blue channel upper threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_upper_thresh: Red channel upper threshold value.
        green_lower_thresh: Green channel lower threshold value.
        blue_lower_thresh: Blue channel lower threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Green", t.elapsed())
    return result


def filter_green_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out green pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    t = Time()
    result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
                     filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
                     filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
                     filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
                     filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
                     filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
                     filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
                     filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
                     filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
                     filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
                     filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
                     filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
                     filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
                     filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
                     filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Green Pen", t.elapsed())
    return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                                display_np_info=False):
    """
    Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
    red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_upper_thresh: Red channel upper threshold value.
        green_upper_thresh: Green channel upper threshold value.
        blue_lower_thresh: Blue channel lower threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Blue", t.elapsed())
    return result


def filter_blue_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out blue pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    t = Time()
    result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
                     filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
                     filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
                     filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
                     filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
                     filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
                     filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
                     filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
                     filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
                     filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
                     filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
                     filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Blue Pen", t.elapsed())
    return result


def filter_grays(rgb, tolerance=15, output_type="bool"):
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.

    Args:
        np_img: RGB image as a NumPy array.
        tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    t = Time()
    (h, w, c) = rgb.shape

    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Grays", t.elapsed())
    return result





def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
 
    t = Time()

    rem_sm = np_img.astype(bool)    # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = min_size / 2
        print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
            mask_percentage, overmask_thresh, min_size, new_min_size))
        rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255
 
    return np_img



 






def save_image(np_img, slide_num, filter_num, file_text):
 
    mask_percentage = None
    # if display_mask_percentage:
    #     mask_percentage = mask_percent(np_img)
    #     display_text = display_text + "\n(" + mask_percentage_text(mask_percentage) + " masked)"
    if slide_num is None and filter_num is None:
        pass
    elif filter_num is None:
        display_text = "S%03d " % slide_num + display_text
    elif slide_num is None:
        display_text = "F%03d " % filter_num + display_text
    else:
        display_text = "S%03d-F%03d " % (slide_num, filter_num) + display_text
    if display:
        display_img(np_img, display_text)
    if save:
        save_filtered_image(np_img, slide_num, filter_num, file_text)
    if info is not None:
        info[slide_num * 1000 + filter_num] = (slide_num, filter_num, display_text, file_text, mask_percentage)



def save_filtered_image(np_img, base_addr, slide_name,fliter_number,filter_text,type_ = '.png'):
 
    t = Time()
    pil_img = np_to_pil(np_img)
    filepath = os.path.join(base_addr,"filter",slide_name + '-' + str(fliter_number)+filter_text+type_)
    
    pil_img.save(filepath)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), filepath))



def apply_image_filters(np_img, base_addr,slide_name  ): 
    rgb = np.array( np_img) 
    len_ = rgb.shape[0] 
    rgb[:int(len_/3),:,:] = 0
    rgb[int(2*len_/3):,:,:] = 0
    
    # save_filtered_image(rgb,base_addr, slide_name,1, "-".join(['-Original',"rgb-"]))

    mask_not_green = filter_green_channel(rgb)
    rgb_not_green = mask_rgb(rgb, mask_not_green)
    # save_filtered_image( rgb_not_green, base_addr,slide_name, 2, 
    #                     "-".join(['-Not Green',"rgb-not-green-"]))
 
    mask_not_gray = filter_grays(rgb)
    rgb_not_gray = mask_rgb(rgb, mask_not_gray)
    # save_filtered_image(rgb_not_gray, base_addr,slide_name, 3, 
    #                     "-".join(['-Not Gray',"rgb-not-gray-"])) 

    mask_no_red_pen = filter_red_pen(rgb)
    rgb_no_red_pen = mask_rgb(rgb, mask_no_red_pen)
    # save_filtered_image(rgb_no_red_pen, base_addr,slide_name, 4,
    #                     "-".join(['-No Red Pen',"rgb-no-red-pen-"]))  

    mask_no_green_pen = filter_green_pen(rgb)
    rgb_no_green_pen = mask_rgb(rgb, mask_no_green_pen)
    # save_filtered_image(rgb_no_green_pen, base_addr,slide_name, 5, 
    #                     "-".join(['-No Green Pen',"rgb-no-green-pen-"]))  

    mask_no_blue_pen = filter_blue_pen(rgb)
    rgb_no_blue_pen = mask_rgb(rgb, mask_no_blue_pen)
    # save_filtered_image(rgb_no_blue_pen, base_addr,slide_name, 6, 
    #                     "-".join(['-No Blue Pen',"rgb-no-blue-pen-"]))   
 
    
    # rgb_filter_kmeans = mask_rgb(rgb, mask_filter_kmeans[:,:,3])
    # save_filtered_image(rgb_filter_kmeans, base_addr,slide_name, 6, 
    #                     "-".join(['-filter_kmeans',"filter_kmeans03-"]))   
    
     
    rgb_no_blue_pen = mask_rgb(rgb, mask_no_blue_pen)
    # save_filtered_image(rgb_no_blue_pen, base_addr,slide_name, 6, 
    #                     "-".join(['-No Blue Pen',"rgb-no-blue-pen-"]))   
    mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
    rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)
    
    # save_filtered_image(rgb_gray_green_pens, base_addr,slide_name, 7, 
    #                     "-".join(['-Not Gray, Not Green, No Pens', 
    #                               "rgb-no-gray-no-green-no-pens-"])) 
    # k-means mask spot
# =============================================================================
#     fill_coins = ndi.binary_fill_holes(rgb_gray_green_pens[:,:,0]/255.)
#     # mask_k_mean = np.zeros_like(fill_coins)
#     # plt.imshow(fill_coins)
#     k_mean_results = slic(rgb_gray_green_pens, 
#                         n_segments=10, 
#                         compactness=1,
#                         start_label =1 ,
#                         multichannel = True,
#                         enforce_connectivity = False,
#                         mask = fill_coins ) 
#     # plt.imshow(k_mean_results[14700:17500,3500:1  1000]) 
#     k_mean_results[k_mean_results>2] =0 
#     # plt.imshow(k_mean_results[14700:17500,3500:11000]) 
#     
#     mask_gray_green_pens2 = k_mean_results& mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
#     rgb_gray_green_pens2 = mask_rgb(rgb, mask_gray_green_pens2)
# =============================================================================
    # plt.imshow(rgb_gray_green_pens2[14700:17500,3500:11000]) 
    
    # save_filtered_image(np.uint8(rgb_gray_green_pens2), base_addr,slide_name, 8, 
    #                     "-".join(['-Not Gray, Not Green, No Pens', 
    #                               "rgb-no-gray-no-green-no-pens-K-means"])) 
        
        
 
    # mask_remove_small = filter_remove_small_objects(mask_gray_green_pens2, 
    #                                                 min_size=500, 
    #                                                 output_type="bool")
    
    # rgb_remove_final = mask_rgb(rgb, mask_remove_small)
    # plt.imshow(k_mean_results[14700:17500,4500:11000]) 
    # rgb_remove_all_combine = mask_rgb(rgb, k_mean_results) 
    # plt.imshow(rgb_remove_all_combine[14700:17500,4500:11000]) 
    # plt.imshow(rgb_remove_final[14700:17500,4500:11000,:]) 
    
    # img = rgb_remove_final 
    img = np.uint8(rgb_gray_green_pens)
    
    return(img)


# filename = temp_[1]
def parse_dimensions_from_image_filename(filename):
 
    m = re.match(".*-r-([\d]*)-r-([\d]*)-n-([\d]*)-n-([\d]*)", filename)
    large_w = int(m.group(1))
    large_h = int(m.group(2))
    small_w = int(m.group(3))
    small_h = int(m.group(4))
    return large_w, large_h, small_w, small_h
    

#  
def apply_filters_to_image(slide_path_list, base_addr  ):
     
    path_fliter = os.path.join(base_addr,'filter')
    if  os.path.exists(path_fliter)==False:
        os.makedirs(path_fliter)
     
    count_ =1
    for i_file in slide_path_list:
        t = Time()
        print("Processing slide " + i_file) 
        np_orig = open_image_np(i_file)
        temp_file = os.path.split(i_file)[1]
        file_name = temp_file.split("-r-")[0] 
        o_h,o_w,n_h,n_w = parse_dimensions_from_image_filename(temp_file)
        temp_filtered_imge = apply_image_filters(np_img=np_orig, 
                                                 base_addr=base_addr,
                                                 slide_name=file_name)
        save_filtered_image(temp_filtered_imge, base_addr, file_name,  0, 
                           "-".join(["-r",str(o_h),"r",str(o_w),
                                     "n",str(n_h),"n",str(n_w),
                                     "filtered_",str(count_)]))
        count_ = count_+1
        
        
# =============================================================================
# # tiles
# =============================================================================
def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
    """
    Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
    a column tile size.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        row_tile_size: Number of pixels in a tile row.
        col_tile_size: Number of pixels in a tile column.

    Returns:
        Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
        into given the row tile size and the column tile size.
    """
    num_row_tiles = math.ceil(rows / row_tile_size)
    num_col_tiles = math.ceil(cols / col_tile_size)
    return num_row_tiles, num_col_tiles

def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

    Args:
        np_img: Image as a NumPy array.

    Returns:
        The percentage of the NumPy array that is masked.
    """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage

def tissue_percent(np_img):

    return 100 - mask_percent(np_img)







class TileSummary:
    """
    Class for tile summary information.
    """

    slide_num = 0
    orig_w = None
    orig_h = None
    orig_tile_w = None
    orig_tile_h = None
    scale_factor = None
    scaled_w = None
    scaled_h = None
    scaled_tile_w = None
    scaled_tile_h = None
    mask_percentage = None
    num_row_tiles = None
    num_col_tiles = None

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0

    def __init__(self, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h, scaled_tile_w,
                             scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
        self.slide_num = None
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.orig_tile_w = orig_tile_w
        self.orig_tile_h = orig_tile_h
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h
        self.scaled_tile_w = scaled_tile_w
        self.scaled_tile_h = scaled_tile_h
        self.tissue_percentage = tissue_percentage
        self.num_col_tiles = num_col_tiles
        self.num_row_tiles = num_row_tiles
        self.tiles = []
        self.summary_save_path = ""

    def __str__(self):
        return summary_title(self)  #+ "\n" + summary_stats(self)

    def mask_percentage(self):
        """
        Obtain the percentage of the slide that is masked.

        Returns:
             The amount of the slide that is masked as a percentage.
        """
        return 100 - self.tissue_percentage

    def num_tiles(self):
        """
        Retrieve the total number of tiles.

        Returns:
            The total number of tiles (number of rows * number of columns).
        """
        return self.num_row_tiles * self.num_col_tiles

    def tiles_by_tissue_percentage(self):
        """
        Retrieve the tiles ranked by tissue percentage.

        Returns:
             List of the tiles ranked by tissue percentage.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
        return sorted_list

    def tiles_by_score(self):
        """
        Retrieve the tiles ranked by score.

        Returns:
             List of the tiles ranked by score.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
        return sorted_list

    def top_tiles(self,NUM_TOP_TILES):
        """
        Retrieve the top-scoring tiles.

        Returns:
              List of the top-scoring tiles.
        """
        sorted_tiles = self.tiles_by_score()
        top_tiles_ = sorted_tiles[:NUM_TOP_TILES]
        return top_tiles_

    def get_tile(self, row, col):
        """
        Retrieve tile by row and column.

        Args:
            row: The row
            col: The column

        Returns:
            Corresponding Tile object.
        """
        tile_index = (row - 1) * self.num_col_tiles + (col - 1)
        tile = self.tiles[tile_index]
        return tile



def get_tile_indices(rows, cols, row_tile_size, col_tile_size):
    """
    Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).

    Args:
        rows: Number of rows.
        cols: Number of columns.
        row_tile_size: Number of pixels in a tile row.
        col_tile_size: Number of pixels in a tile column.

    Returns:
        List of tuples representing tile coordinates consisting of starting row, ending row,
        starting column, ending column, row number, column number.
    """
    indices = list()
    num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    for r in range(0, num_row_tiles):
        start_r = r * row_tile_size
        end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
        for c in range(0, num_col_tiles):
            start_c = c * col_tile_size
            end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
            indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
    return indices



class TissueQuantity(Enum):
  NONE = 0
  LOW = 1
  MEDIUM = 2
  HIGH = 3



def tissue_quantity(tissue_percentage,TISSUE_HIGH_THRESH=80,TISSUE_LOW_THRESH=10):
    """
    Obtain TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE) for corresponding tissue percentage.

    Args:
        tissue_percentage: The tile tissue percentage.

    Returns:
        TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE).
    """
    if tissue_percentage >= TISSUE_HIGH_THRESH:
        return TissueQuantity.HIGH
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        return TissueQuantity.MEDIUM
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        return TissueQuantity.LOW
    else:
        return TissueQuantity.NONE



def small_to_large_mapping(small_pixel, large_dimensions,SCALE_FACTOR):
    """
    Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.

    Args:
        small_pixel: The scaled-down width and height.
        large_dimensions: The width and height of the original whole-slide image.

    Returns:
        Tuple consisting of the scaled-up width and height.
  """
    small_x, small_y = small_pixel
    large_w, large_h = large_dimensions
    large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
    large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
    return large_x, large_y


def score_tile(np_tile, tissue_percent, row, col):
 
    color_factor = hsv_purple_pink_factor(np_tile)
    s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
    amount = tissue_quantity(tissue_percent)
    quantity_factor = tissue_quantity_factor(amount)
    combined_factor = color_factor * s_and_v_factor * quantity_factor
    score = (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0
    # scale score to between 0 and 1
    score = 1.0 - (10.0 / (10.0 + score))
    return score, color_factor, s_and_v_factor, quantity_factor



# Filter_file = slide_path_list[1]

def score_tiles(Filter_file, TILE_SIZE=1024, HSV_PURPLE = 270,HSV_PINK=330,
                SCALE_FACTOR=32,small_tile_in_tile=False):
    temp_ = os.path.split(Filter_file)    
    
    o_w, o_h, w, h = parse_dimensions_from_image_filename(temp_[1])

    np_img =  open_image_np(Filter_file)

    row_tile_size = round(TILE_SIZE / SCALE_FACTOR)  # use round?
    col_tile_size = round(TILE_SIZE / SCALE_FACTOR)  # use round?

    num_row_tiles, num_col_tiles = get_num_tiles(h, w, row_tile_size, col_tile_size)

    tile_sum = TileSummary( orig_w=o_w,
                         orig_h=o_h,
                         orig_tile_w=TILE_SIZE,
                         orig_tile_h=TILE_SIZE,
                         scaled_w=w,
                         scaled_h=h,
                         scaled_tile_w=col_tile_size,
                         scaled_tile_h=row_tile_size,
                         tissue_percentage=tissue_percent(np_img),
                         num_col_tiles=num_col_tiles,
                         num_row_tiles=num_row_tiles)

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0
    tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size)
    for t in tile_indices:
        count += 1    # tile_num
        r_s, r_e, c_s, c_e, r, c = t
        np_tile = np_img[r_s:r_e, c_s:c_e]
        t_p = tissue_percent(np_tile)
        amount = tissue_quantity(t_p)
        if amount == TissueQuantity.HIGH:
            high += 1
        elif amount == TissueQuantity.MEDIUM:
            medium += 1
        elif amount == TissueQuantity.LOW:
            low += 1
        elif amount == TissueQuantity.NONE:
            none += 1
        o_c_s, o_r_s = small_to_large_mapping((c_s, r_s), (o_w, o_h),SCALE_FACTOR)
        o_c_e, o_r_e = small_to_large_mapping((c_e, r_e), (o_w, o_h),SCALE_FACTOR)

        # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
        if (o_c_e - o_c_s) > TILE_SIZE:
            o_c_e -= 1
        if (o_r_e - o_r_s) > TILE_SIZE:
            o_r_e -= 1

        score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, r, c)

        np_scaled_tile = np_tile if small_tile_in_tile else None
        tile = Tile(tile_sum, np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                                o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score)
        tile_sum.tiles.append(tile)

    tile_sum.count = count
    tile_sum.high = high
    tile_sum.medium = medium
    tile_sum.low = low
    tile_sum.none = none

    tiles_by_score = tile_sum.tiles_by_score()
    rank = 0
    for t in tiles_by_score:
        rank += 1
        t.rank = rank
    return tile_sum




 





# slide_path_list =  get_slide_path(r'F:\Phase1_Module\Data\Temp_Scale_summary_tiles\filter')
# filter_image_path = slide_path_list[1]

def summary_and_tiles(filter_image_path,SCALE_FACTOR ,TILE_SIZE,
                     NUM_TOP_TILES  , HSV_PURPLE,
                     HSV_PINK, TISSUE_HIGH_THRESH, TISSUE_LOW_THRESH,
                      save_summary=True ):
 
    tile_sum = score_tiles(filter_image_path, 
                           TILE_SIZE=TILE_SIZE, 
                           HSV_PURPLE = HSV_PURPLE,
                           HSV_PINK=HSV_PINK,
                           SCALE_FACTOR=SCALE_FACTOR)
    temp_0 = os.path.split(filter_image_path)
    temp_1 = os.path.split(temp_0[0])
    temp_2 = os.path.join(temp_1[0],"tiles_summary")
    slide_name = temp_0[1].split("-0-r")[0]
    
    if not os.path.exists(temp_2):
        os.makedirs(temp_2)
    file_save_name = os.path.join(temp_2 ,re.sub('.png',".csv",temp_0[1]))
    tile_sum.summary_save_path = file_save_name
    if save_summary:
        save_tile_data(tile_sum)
        
    # generate_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
    # generate_top_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
    # if save_top_tiles:
    #     for tile in tile_sum.top_tiles():
    #         tile.save_tile(base_addr=temp_1[0],slide_name=slide_name)
    return tile_sum

# =============================================================================
# 
# =============================================================================

def hsv_purple_pink_factor(rgb):
    """
    Compute scoring factor based on purple and pink HSV hue deviations and degree to which a narrowed hue color range
    average is purple versus pink.

    Args:
        rgb: Image an NumPy array.

    Returns:
        Factor that favors purple (hematoxylin stained) tissue over pink (eosin stained) tissue.
    """
    hues = rgb_to_hues(rgb)
    hues = hues[hues >= 260]    # exclude hues under 260
    hues = hues[hues <= 340]    # exclude hues over 340
    if len(hues) == 0:
        return 0    # if no hues between 260 and 340, then not purple or pink
    pu_dev = hsv_purple_deviation(hues)
    pi_dev = hsv_pink_deviation(hues)
    avg_factor = (340 - np.average(hues)) ** 2

    if pu_dev == 0:    # avoid divide by zero if tile has no tissue
        return 0

    factor = pi_dev / pu_dev * avg_factor
    return factor

def filter_rgb_to_hsv(np_img, display_np_info=True):
    """
    Filter RGB channels to HSV (Hue, Saturation, Value).

    Args:
        np_img: RGB image as a NumPy array.
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        Image as NumPy array in HSV representation.
    """

    if display_np_info:
        t = Time()
    hsv = sk_color.rgb2hsv(np_img)
    if display_np_info:
        np_info(hsv, "RGB to HSV", t.elapsed())
    return hsv

def filter_hsv_to_h(hsv, output_type="int", display_np_info=True):
    """
    Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
    values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
    https://en.wikipedia.org/wiki/HSL_and_HSV

    Args:
        hsv: HSV image as a NumPy array.
        output_type: Type of array to return (float or int).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        Hue values (float or int) as a 1-dimensional NumPy array.
    """
    if display_np_info:
        t = Time()
    h = hsv[:, :, 0]
    h = h.flatten()
    if output_type == "int":
        h *= 360
        h = h.astype("int")
    if display_np_info:
        np_info(hsv, "HSV to H", t.elapsed())
    return h



def rgb_to_hues(rgb):
    """
    Convert RGB NumPy array to 1-dimensional array of hue values (HSV H values in degrees).

    Args:
        rgb: RGB image as a NumPy array

    Returns:
        1-dimensional array of hue values in degrees
    """
    hsv = filter_rgb_to_hsv(rgb, display_np_info=False)
    h = filter_hsv_to_h(hsv, display_np_info=False)
    return h



def hsv_saturation_and_value_factor(rgb):
    """
    Function to reduce scores of tiles with narrow HSV saturations and values since saturation and value standard
    deviations should be relatively broad if the tile contains significant tissue.

    Example of a blurred tile that should not be ranked as a top tile:
        ../data/tiles_png/006/TUPAC-TR-006-tile-r58-c3-x2048-y58369-w1024-h1024.png

    Args:
        rgb: RGB image as a NumPy array

    Returns:
        Saturation and value factor, where 1 is no effect and less than 1 means the standard deviations of saturation and
        value are relatively small.
    """
    hsv = filter_rgb_to_hsv(rgb, display_np_info=False)
    s = filter_hsv_to_s(hsv)
    v = filter_hsv_to_v(hsv)
    s_std = np.std(s)
    v_std = np.std(v)
    if s_std < 0.05 and v_std < 0.05:
        factor = 0.4
    elif s_std < 0.05:
        factor = 0.7
    elif v_std < 0.05:
        factor = 0.7
    else:
        factor = 1

    factor = factor ** 2
    return factor


def filter_hsv_to_s(hsv):
    """
    Experimental HSV to S (saturation).

    Args:
        hsv:    HSV image as a NumPy array.

    Returns:
        Saturation values as a 1-dimensional NumPy array.
    """
    s = hsv[:, :, 1]
    s = s.flatten()
    return s


def filter_hsv_to_v(hsv):
    """
    Experimental HSV to V (value).

    Args:
        hsv:    HSV image as a NumPy array.

    Returns:
        Value values as a 1-dimensional NumPy array.
    """
    v = hsv[:, :, 2]
    v = v.flatten()
    return v


def tissue_quantity_factor(amount):
    """
    Obtain a scoring factor based on the quantity of tissue in a tile.

    Args:
        amount: Tissue amount as a TissueQuantity enum value.

    Returns:
        Scoring factor based on the tile tissue quantity.
    """
    if amount == TissueQuantity.HIGH:
        quantity_factor = 1.0
    elif amount == TissueQuantity.MEDIUM:
        quantity_factor = 0.2
    elif amount == TissueQuantity.LOW:
        quantity_factor = 0.1
    else:
        quantity_factor = 0.0
    return quantity_factor

 
class Tile:
    """
    Class for information about a tile.
    """

    def __init__(self, tile_summary, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                             o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
        self.tile_summary = tile_summary
        # self.slide_num = slide_num
        self.Raw_file_path = None
        self.np_scaled_tile = np_scaled_tile
        self.tile_num = tile_num
        self.r = r
        self.c = c
        self.r_s = r_s
        self.r_e = r_e
        self.c_s = c_s
        self.c_e = c_e
        self.o_r_s = o_r_s
        self.o_r_e = o_r_e
        self.o_c_s = o_c_s
        self.o_c_e = o_c_e
        self.tissue_percentage = t_p
        self.color_factor = color_factor
        self.s_and_v_factor = s_and_v_factor
        self.quantity_factor = quantity_factor
        self.score = score

    def __str__(self):
        return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
            self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

    def __repr__(self):
        return "\n" + self.__str__()

    def mask_percentage(self):
        return 100 - self.tissue_percentage

    def tissue_quantity(self):
        return tissue_quantity(self.tissue_percentage)

    def get_pil_tile(self):
        return tile_to_pil_tile(self)

    # def get_np_tile(self):
    #     return tile_to_np_tile(self)

    def save_tile(self,base_addr ):
        save_display_tile(self, base_addr,
                          self.Raw_file_path,
                          save=True )

  # def display_tile(self):
  #   save_display_tile(self, save=False)

    # def display_with_histograms(self):
    #     display_tile(self, rgb_histograms=True, hsv_histograms=True)

    def get_np_scaled_tile(self):
        return self.np_scaled_tile

    def get_pil_scaled_tile(self):
        return np_to_pil(self.np_scaled_tile)




def hsv_purple_deviation(hsv_hues,HSV_PURPLE = 270):
    """
    Obtain the deviation from the HSV hue for purple.

    Args:
        hsv_hues: NumPy array of HSV hue values.

    Returns:
        The HSV purple deviation.
    """
    purple_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PURPLE) ** 2))
    return purple_deviation



def hsv_pink_deviation(hsv_hues,HSV_PINK=330):
    """
    Obtain the deviation from the HSV hue for pink.

    Args:
        hsv_hues: NumPy array of HSV hue values.

    Returns:
        The HSV pink deviation.
    """
    pink_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PINK) ** 2))
    return pink_deviation


def hsv_purple_pink_factor(rgb):
    """
    Compute scoring factor based on purple and pink HSV hue deviations and degree to which a narrowed hue color range
    average is purple versus pink.

    Args:
        rgb: Image an NumPy array.

    Returns:
        Factor that favors purple (hematoxylin stained) tissue over pink (eosin stained) tissue.
    """
    hues = rgb_to_hues(rgb)
    hues = hues[hues >= 260]    # exclude hues under 260
    hues = hues[hues <= 340]    # exclude hues over 340
    if len(hues) == 0:
        return 0    # if no hues between 260 and 340, then not purple or pink
    pu_dev = hsv_purple_deviation(hues)
    pi_dev = hsv_pink_deviation(hues)
    avg_factor = (340 - np.average(hues)) ** 2

    if pu_dev == 0:    # avoid divide by zero if tile has no tissue
        return 0

    factor = pi_dev / pu_dev * avg_factor
    return factor



def summary_title(tile_summary):
    """
    Obtain tile summary title.

    Args:
        tile_summary: TileSummary object.

    Returns:
         The tile summary title.
    """
    return "Slide Tile Summary below" 


 

# tile_summary = tile_sum
def save_tile_data(tile_summary):
    """
    Save tile data to csv file.

    Args
        tile_summary: TimeSummary object.
    """

    time = Time()

    csv = summary_title(tile_summary)    

    csv += "\n\n\nTile Num,Row,Column,Tissue %,Tissue Quantity,Col Start,Row Start,Col End,Row End,Col Size,Row Size," + \
                 "Original Col Start,Original Row Start,Original Col End,Original Row End,Original Col Size,Original Row Size," + \
                 "Color Factor,S and V Factor,Quantity Factor,Score\n"

    for t in tile_summary.tiles:
        line = "%d,%d,%d,%4.2f,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%4.0f,%4.2f,%4.2f,%0.4f\n" % (
            t.tile_num, t.r, t.c, t.tissue_percentage, t.tissue_quantity().name, t.c_s, t.r_s, t.c_e, t.r_e, t.c_e - t.c_s,
            t.r_e - t.r_s, t.o_c_s, t.o_r_s, t.o_c_e, t.o_r_e, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s, t.color_factor,
            t.s_and_v_factor, t.quantity_factor, t.score)
        csv += line
 
    csv_file = open(tile_summary.summary_save_path, "w")
    csv_file.write(csv)
    csv_file.close()
 




def tile_to_pil_tile(tile):
    """
    Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.

    Args:
        tile: Tile object.

    Return:
        Tile as a PIL image.
    """
    t = tile
    # slide_filepath = get_training_slide_path(t.slide_num) # raw slide
    # print("*"*60)
    # print(t.Raw_file_path)
    s = openslide.open_slide(tile.Raw_file_path)

    x, y = t.o_c_s, t.o_r_s
    w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s
    tile_region = s.read_region((x, y), 0, (w, h))
    # RGBA to RGB
    pil_img = tile_region.convert("RGB")
    return pil_img

def get_tile_image_path(tile,base_addr,slide_name,type_ = 'png'):
    """
    Obtain tile image path based on tile information such as row, column, row pixel position, column pixel position,
    pixel width, and pixel height.

    Args:
        tile: Tile object.

    Returns:
        Path to image tile.
    """
    t = tile
    tile_path = os.path.join(base_addr, "tiles",
                                                     slide_name + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
                                                         t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s) + "." + type_)
    return tile_path



def save_display_tile(tile, base_addr,Raw_slide_file,save=True ):
    """
    Save and/or display a tile image.

    Args:
        tile: Tile object.
        save: If True, save tile image.
        display: If True, dispaly tile image.
    """
    tile.Raw_file_path = Raw_slide_file
    tile_pil_img = tile_to_pil_tile(tile)
    file_full_name = os.path.split(Raw_slide_file)[1]

    if save:
        t = Time()
        
        img_path = get_tile_image_path(tile,base_addr,
                                       file_full_name.split(".")[0])
        dir_ = os.path.join(base_addr,"tiles")
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        tile_pil_img.save(img_path)
        print("%-20s | Time: %-14s    Name: %s" % ("Save Tile", str(t.elapsed()), img_path))
 
        
        
              
        
def search_raw_file_path(search_list,search_file):
    temp_0 = list(map(lambda x:os.path.split(x)[1],search_list))
    temp_1 = list(map(lambda x:re.search(search_file,x)!=None,temp_0 ))
    search_list = pd.DataFrame(search_list) 
    return search_list[temp_1].iloc[0,0]


 

class Slide_combine_fliter(object):
    def __init__(self,RAW_SLIDE_FILE_LIST, unique_id,BASE_ADDR,SCALE_FACTOR = 16,
                        TISSUE_HIGH_THRESH = 80, TISSUE_LOW_THRESH = 10,
                        TILE_SIZE = 512, NUM_TOP_TILES = 1000,
                        HSV_PURPLE = 270, HSV_PINK = 330):
        self.RAW_SLIDE_FILE_LIST = RAW_SLIDE_FILE_LIST
        self.BASE_ADDR = BASE_ADDR
        self.SCALE_FACTOR = SCALE_FACTOR
        self.TISSUE_HIGH_THRESH = TISSUE_HIGH_THRESH
        self.TISSUE_LOW_THRESH = TISSUE_LOW_THRESH
        self.TILE_SIZE = TILE_SIZE
        self.NUM_TOP_TILES = NUM_TOP_TILES
        self.HSV_PURPLE = HSV_PURPLE
        self.HSV_PINK = HSV_PINK
        
        self.unique_id = unique_id
        
    def Scale_slide(self):
        slide_range_to_images(self.RAW_SLIDE_FILE_LIST, 
                              self.SCALE_FACTOR,
                              self.BASE_ADDR,
                              self.unique_id) 
    
    def fliter_slide(self,Scaled_image_list):
        apply_filters_to_image(Scaled_image_list,
                               self.BASE_ADDR)
        self.Scaled_image_list = Scaled_image_list
        
    def Tiles_save(self,Flitered_image_list,save_tiles = True): 
        for i in range(len(Flitered_image_list)):
            # pass
            tile_sum = summary_and_tiles(filter_image_path=Flitered_image_list[i], 
                              TILE_SIZE = self.TILE_SIZE,
                              SCALE_FACTOR=self.SCALE_FACTOR,
                              NUM_TOP_TILES = self.NUM_TOP_TILES,
                              HSV_PURPLE=self.HSV_PURPLE,
                              HSV_PINK=self.HSV_PINK,
                              TISSUE_HIGH_THRESH=self.TISSUE_HIGH_THRESH,
                              TISSUE_LOW_THRESH=self.TISSUE_LOW_THRESH)
            if(save_tiles):
                Raw_file = os.path.split(Flitered_image_list[i])
                Raw_file = Raw_file[1].split("-0-r-")[0]
                Raw_file_i = search_raw_file_path( self.RAW_SLIDE_FILE_LIST,Raw_file)
                tile_sum.Raw_file_path = Raw_file_i     
                for tile in tile_sum.top_tiles(self.NUM_TOP_TILES):
                    save_display_tile( tile, self.BASE_ADDR,
                               Raw_slide_file = Raw_file_i,
                              save=True )
                
                 
                    
def get_tiles_res(Raw_file_addr,
                  tile_summary_addr,
                  seed_in,extrect_num,
                  unique_sample_id ,
                  tiles_out_addr):
    summary_file = pd.read_csv(tile_summary_addr,skiprows=1)
    summary_file = summary_file[summary_file['Score']!=0] 
    temp_des = summary_file['Score'].describe()
    summary_file = summary_file[summary_file['Score']>=temp_des[4]] 
    random.seed(seed_in)
    All_tiles = summary_file.shape[0]
    summary_file = summary_file.reset_index(drop=True)
    index_tiles = list(range(All_tiles))
    random.shuffle( index_tiles)
    if extrect_num>=All_tiles:
        need_tile_index = index_tiles
        
    else:
        need_tile_index = index_tiles[:extrect_num]
     
    Raw_slide = openslide.open_slide(Raw_file_addr)
    for i_tile in need_tile_index:
        temp_need = summary_file.iloc[i_tile,:]
        x, y = temp_need[11], temp_need[12]
        w, h = temp_need[13] - temp_need[11], temp_need[14] - temp_need[12]
        tile_region = Raw_slide.read_region((x, y), 0, (w, h))
        # RGBA to RGB
        pil_img = tile_region.convert("RGB")
        File_tile_i = unique_sample_id + '_num_'+str(temp_need[0]) + '.png'
        file_name_tile = os.path.join(tiles_out_addr,File_tile_i)
        pil_img.save(file_name_tile)
    Raw_slide.close()
        
        
        
        
        
        
        
        
## test part
# =============================================================================
# temp_slide = get_slide_path( r'E:\MPN\Raw_data\ywei20200916',r"mrxs")
# img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_adr=temp_slide[1])
# # img.show()
# rgb = pil_to_np_rgb(img)
# grayscale = filter_rgb_to_grayscale(rgb)
# display_img(grayscale, "Grayscale")
# # complement
# complement1 = filter_complement(grayscale)
# display_img(complement1, "Complement")
# # base threshold 
# thresh = filter_threshold(complement1, threshold=100)
# display_img(thresh, "Threshold")
# # hysteresis threshold
# thresh = filter_hysteresis_threshold(complement1 )
# display_img(thresh, "Threshold")
# # Otsu Threshold
# thresh = filter_otsu_threshold(complement1)
# display_img(thresh, "Threshold")
# =============================================================================

# ########  whole process
# file_name_list = get_slide_path( r'E:\MPN\Raw_data\ywei20200916',r"mrxs")
# base_addr = r'E:\MPN\Raw_data\test_own_code'
# # step1 scale
# slide_range_to_images(file_name_list, 32,base_addr)
# # step2 filter    
# slide_path_list = get_slide_path( r'E:\MPN\Raw_data\test_own_code\scale',r"1M")
# apply_filters_to_image(slide_path_list,base_addr)
# # step3 tile score 
# slide_path_list = get_slide_path( r'E:\MPN\Raw_data\test_own_code\filter',r"filtered")
# # filter_image_path = slide_path_list[1]
# summary_and_tiles( filter_image_path=slide_path_list[1], SCALE_FACTOR=32)

# for i in range(len(slide_path_list)):
#     summary_and_tiles(slide_path_list[i],SCALE_FACTOR=32)
 

# =============================================================================
# # test whole
# file_name_list = get_slide_path( r'E:\MPN\Raw_data\ywei20200916',r"mrxs")
# base_addr = r'E:\MPN\Raw_data\test_own_code'
# test_whole = Slide_combine_fliter( file_name_list,base_addr)
# ## step1
# test_whole.Scale_slide()                        ## 生成小图片
# ## step2
# slide_path_list = get_slide_path( r'E:\MPN\Raw_data\test_own_code\scale',r"1M")
# test_whole.fliter_slide(slide_path_list)        ## 生成fliter mask
# ## step3                                        ## score and tiles
# slide_path_list = get_slide_path( r'E:\MPN\Raw_data\test_own_code\filter',r"filtered")
# test_whole.Tiles_save(slide_path_list ) 
# =============================================================================

 

