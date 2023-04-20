import logging
import SimpleITK as sitk
from typing import Tuple
import numpy as np
import os
from rt_utils import RTStruct
from typing import Optional, Union, List
from shutil import copyfile
import time

def axes_swapping(array: np.array):
    array = array.T
    array = np.swapaxes(array, 0, 1)
    # array = array[:,::-1,:]
    return array

def itk_resample_volume(
    img: sitk.Image, 
    new_spacing : tuple,
    interpolator = sitk.sitkLinear
    ) -> sitk.Image:

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    args = [img, new_size, sitk.Transform(), interpolator,
            img.GetOrigin(), new_spacing, img.GetDirection(), 0,
            # 3]
            img.GetPixelID()]
    return sitk.Resample(*args)

# def load_nifti_pred(nifti_pred_file: str, new_spacing: tuple = (1.0, 1.0, 1.0)):
#     '''Loads nifti file with predictions using sitk and rotates it so that it matches dicom orientation.
    
#     Args:
#         nifti_file (str): Path to nifti file
#         new_spacing (tuple): The desired spatial spacing of the voxels. DeepMedic has been trained using a resolution of (1,1,1)'''
#     nifti_pred = sitk.ReadImage(nifti_pred_file)
#     nifti_pred = itk_resample_volume(nifti_pred, new_spacing=new_spacing) # this just resamples the image
#     nifti_pred_data = axes_swapping(sitk.GetArrayFromImage(nifti_pred))
#     return nifti_pred_data

def find_clusters_itk(mask: sitk.Image, max_object: int = 50) -> Tuple[sitk.Image, int]:
    """
    Find how many clusters in the mask (ITK Image).
    Args:
        mask: a binary sitk.Image
        max_object: a threshold to make sure the # of object does not exceed max_object.
    Return:
        label_image: a masked sitk.Image with cluster index
        num: number of clusters
    """

    # Make sure mask is a binary image
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Make sure they are binary
    min_max_f = sitk.MinimumMaximumImageFilter()
    min_max_f.Execute(mask)
    mask_max = min_max_f.GetMaximum()
    mask_min = min_max_f.GetMinimum()

    assert mask_max <= 1, f"mask should be binary, but the maximum value is {mask_max}"
    assert mask_min >= 0, f"mask should be binary, but the minimum value is {mask_min}"

    label_map_f = sitk.BinaryImageToLabelMapFilter()
    label_map = label_map_f.Execute(mask)

    label_map_to_image_f = sitk.LabelMapToLabelImageFilter()
    label_image = label_map_to_image_f.Execute(label_map)

    num = label_map_f.GetNumberOfObjects()

    assert (
        num <= max_object
    ), f"Does not expect there are clusters more than {max_object} but is {num}"

    return label_image, num

def get_binary_rtss(nifti_pred: str, inference_threshold: float, new_spacing: tuple = None):

    rtss = sitk.ReadImage(nifti_pred)

    if new_spacing is not None:
        rtss = itk_resample_volume(rtss, new_spacing) # resamples data to original dicom resolution before executing binary threshold
    
    binary_filter = sitk.BinaryThresholdImageFilter()
    binary_filter.SetLowerThreshold(inference_threshold)
    binary_filter.SetUpperThreshold(1)

    rtss_binary = binary_filter.Execute(rtss)

    return rtss_binary

def copy_file_safely(tmp_dir: str, src: str, dst_naming: str) -> str:
    """Make sure file will not overwrite another file"""
    dst = None

    try:
        dst = os.path.join(tmp_dir, dst_naming)

        if os.path.isfile(dst):
            dst = dst + time.strftime("%H%M%S")
        copyfile(src, dst)

    except Exception as e:
        print(e)

    return dst

def fetch_all_rois(rtstruct: RTStruct) -> np.ndarray:

    masks = []
    roi_names = rtstruct.get_roi_names()

    for roi_name in roi_names:
        mask = rtstruct.get_roi_mask_by_name(roi_name)
        masks.append(mask)

    if len(masks) == 0:
        return None

    background = np.sum(masks, axis=0) == 0

    masks.insert(0, background)  # Add background to the mask

    return np.stack(masks, axis=-1)