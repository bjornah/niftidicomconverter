import logging
import os
import time

import SimpleITK as sitk
import nibabel as nib
import numpy as np

from typing import Tuple
from rt_utils import RTStruct
from typing import Optional, Union, List
from shutil import copyfile

from niftidicomconverter.dicomhandling import itk_resample_volume

def axes_swapping(array: np.array):
    """
    Swap the x and y axes of a 3D NumPy array.

    Args:
        array (np.array): A 3D NumPy array with shape (z, y, x).

    Returns:
        np.array: A copy of the input array with the x and y axes swapped, and shape (z, x, y).
    """
    array = array.T
    array = np.swapaxes(array, 0, 1)
    # array = array[:,::-1,:]
    return array

def find_clusters_itk(mask: sitk.Image, max_object: int = 50) -> Tuple[sitk.Image, int]:
    """
    Find the number of separate clusters in the mask (ITK Image).
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
    """
    Load a NIfTI file containing an RTSS and apply a binary threshold to create a binary mask.
    Used to convert prediction probabilities to binary classification. This is relevant, e.g.,
    when an autosegmentation algorithm has yielded predictions that have been stored in a nifti
    file format and that have not yet been binarised.

    Args:
        nifti_pred (str): The path to the NIfTI file containing the RTSS to be thresholded.
        inference_threshold (float): The threshold value to use for the binary thresholding.
        new_spacing (Optional[Tuple[float, float, float]]): The desired voxel spacing for the resampled image.
            If provided, the image is resampled to this voxel spacing before thresholding.

    Returns:
        sitk.Image: The binary mask obtained by thresholding the RTSS.

    Raises:
        FileNotFoundError: If the specified NIfTI file does not exist.
    """
    rtss = sitk.ReadImage(nifti_pred)

    if new_spacing is not None:
        rtss = itk_resample_volume(rtss, new_spacing) # resamples data to original dicom resolution before executing binary threshold
    
    binary_filter = sitk.BinaryThresholdImageFilter()
    binary_filter.SetLowerThreshold(inference_threshold)
    binary_filter.SetUpperThreshold(1)

    rtss_binary = binary_filter.Execute(rtss)

    return rtss_binary

def copy_file_safely(tmp_dir: str, src: str, dst_naming: str) -> str:
    """
    Copy a file from the source path to a destination directory, ensuring the destination filename is unique.
    This makes sure you do not overwrite another file.

    Args:
        tmp_dir (str): The path to the temporary directory where the file will be copied.
        src (str): The path to the source file to be copied.
        dst_naming (str): The desired filename for the copied file.

    Returns:
        str: The path to the copied file, including the unique filename suffix if necessary.
    """
    
    dst = None

    try:
        dst = os.path.join(tmp_dir, dst_naming)

        if os.path.isfile(dst):
            dst = dst + time.strftime("%H%M%S")
        copyfile(src, dst)

    except Exception as e:
        print(e)

    return dst


def get_array_from_itk_image(image: sitk.Image) -> np.ndarray:
    """
    Convert a SimpleITK image object to a NumPy array.

    Args:
        image: A SimpleITK image object.

    Returns:
        A NumPy array with the same dimensions and pixel values as the input image.
    """
    return sitk.GetArrayFromImage(image)

def copy_nifti_header(src: nib.Nifti1Image, dst: nib.Nifti1Image) -> nib.Nifti1Image:
    """Copy header from src to dst while perserving the dst data."""
    data = dst.get_fdata()
    return nib.nifti1.Nifti1Image(data, None, header=src.header)


# def itk_resample_volume(
#     img: sitk.Image, 
#     new_spacing : tuple,
#     interpolator = sitk.sitkLinear
#     ) -> sitk.Image:
#     """
#     Resample a SimpleITK image to a new spacing using the specified interpolator.

#     Args:
#         img (sitk.Image): The input SimpleITK image to resample.
#         new_spacing (Tuple[float, float, float]): The new spacing to resample the image to, as a tuple of (x, y, z) values.
#         interpolator (Optional[sitk.InterpolatorEnum]): The interpolation method to use when resampling the image.
#             Default is sitk.sitkLinear.

#     Returns:
#         sitk.Image: The resampled SimpleITK image.
#     """
#     original_spacing = img.GetSpacing()
#     original_size = img.GetSize()
#     new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
#     args = [img, new_size, sitk.Transform(), interpolator,
#             img.GetOrigin(), new_spacing, img.GetDirection(), 0,
#             # 3]
#             img.GetPixelID()]
#     return sitk.Resample(*args)

# def load_nifti_pred(nifti_pred_file: str, new_spacing: tuple = (1.0, 1.0, 1.0)):
#     '''Loads nifti file with predictions using sitk and rotates it so that it matches dicom orientation.
    
#     Args:
#         nifti_file (str): Path to nifti file
#         new_spacing (tuple): The desired spatial spacing of the voxels. DeepMedic has been trained using a resolution of (1,1,1)'''
#     nifti_pred = sitk.ReadImage(nifti_pred_file)
#     nifti_pred = itk_resample_volume(nifti_pred, new_spacing=new_spacing) # this just resamples the image
#     nifti_pred_data = axes_swapping(sitk.GetArrayFromImage(nifti_pred))
#     return nifti_pred_data