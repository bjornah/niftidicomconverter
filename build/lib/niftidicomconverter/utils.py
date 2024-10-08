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
import shutil
import pydicom

import re

from niftidicomconverter.dicomhandling import itk_resample_volume

def check_dtype_sitk(fpath):
    image_sitk = sitk.ReadImage(fpath)
    pixel_id = image_sitk.GetPixelID()
    pixel_id_type_string = sitk.GetPixelIDValueAsString(pixel_id)
    print(f"Current data type: {pixel_id_type_string}")

def correct_nifti_header(src, dst):
    """
    Corrects the header of a dst NIfTI file using another NIfTI src file as a reference.
    Additionally, preserves the data type of the dst NIfTI file.

    Parameters:
    src (str): The path to the source NIfTI file.
    dst (str): The path to the destination NIfTI file.

    Returns:
    Nifti1Image: The corrected NIfTI image with the header updated and data type preserved.
    """
    src_nifti = nib.load(src)
    dst_nifti = nib.load(dst)
    
    data = dst_nifti.get_fdata()
    dtype = dst_nifti.get_data_dtype()
    affine = src_nifti.affine
    header = src_nifti.header
    header.set_data_dtype(dtype)
    
    corrected_image = nib.nifti1.Nifti1Image(data.astype(dtype), affine=affine, header=header)
    
    # print(f'dtype = {dtype}')
    # print(f'corrected_image.get_data_dtype() = {corrected_image.get_data_dtype()}')
    
    return corrected_image

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

def find_clusters_itk(mask: sitk.Image, max_object: int = 100) -> Tuple[sitk.Image, int]:
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

def copy_file_safely(file_path, target_dir):
    """
    Copy a file safely to a target directory.

    Args:
        file_path (str): The path of the file to be copied.
        target_dir (str): The target directory where the file will be copied to.

    Returns:
        None

    Raises:
        None
    """
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if os.path.isfile(file_path):
        file_name = os.path.basename(file_path)
        target_file_path = os.path.join(target_dir, file_name)

        # Handle file name conflicts
        counter = 1
        while os.path.exists(target_file_path):
            name, extension = os.path.splitext(file_name)
            new_file_name = f"{name}_{counter}{extension}"
            target_file_path = os.path.join(target_dir, new_file_name)
            counter += 1

        # Copy the file
        try:
            shutil.copy2(file_path, target_file_path)
            logging.debug(f"Copied {file_path} to {target_file_path}")
        except Exception as e:
            logging.exception(f"Error copying {file_path} to {target_file_path}: {e}")           
        
    else:
        logging.info(f"Skipping non-existent file: {file_path}")


def get_array_from_itk_image(image: sitk.Image) -> np.ndarray:
    """
    Convert a SimpleITK image object to a NumPy array.

    Args:
        image: A SimpleITK image object.

    Returns:
        A NumPy array with the same dimensions and pixel values as the input image.
    """
    return sitk.GetArrayFromImage(image)


def calculate_new_affine(old_affine, old_spacing, new_spacing, dimensions):
    """
    Calculate the new affine matrix after resampling.

    Parameters:
    old_affine (np.array): The original affine matrix.
    old_spacing (tuple): The original spacing.
    new_spacing (tuple): The new spacing.
    dimensions (list): The dimensions that are being resampled.

    Returns:
    np.array: The new affine matrix.
    """
    new_affine = old_affine.copy()
    for dim in dimensions:
        scale = old_spacing[dim] / new_spacing[dim]
        new_affine[dim, :3] *= scale
    return new_affine

def generate_dicom_filename(dicom_file_path):
    """
    Generates a filename from a DICOM file by concatenating the PatientID and SeriesInstanceUID,
    separated by an underscore, and removing any special characters.

    Parameters:
    - dicom_file_path (str): The path to the DICOM file.

    Returns:
    - str: A sanitized string suitable for use as a filename.
    """
    try:
        dicom_data = pydicom.dcmread(dicom_file_path)
        patient_id = getattr(dicom_data, 'PatientID', 'UnknownPatient')
        series_instance_uid = getattr(dicom_data, 'SeriesInstanceUID', 'UnknownSeries')

        # Concatenate with an underscore
        filename = f"{patient_id}_{series_instance_uid}"

        # Remove special characters
        sanitized_filename = re.sub(r'[^a-zA-Z0-9_]', '', filename)

        return sanitized_filename
    except Exception as e:
        logging.exception(f"Error processing DICOM file: {e}")
        return None

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