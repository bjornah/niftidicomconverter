import os
import re

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import pandas as pd

import pydicom
import tempfile
import logging
import glob

from tqdm import tqdm
from natsort import natsorted
from typing import Tuple, Optional, List, Union, Dict, Any
from collections import defaultdict
from pydicom.errors import InvalidDicomError

def filter_dicom_files(subdirectory: str, glob_pattern: Optional[str], existing_database_df: Optional[pd.DataFrame]) -> List[str]:
    """
    Filters out already processed DICOM files based on the existing database.

    Args:
        subdirectory (str): The current subdirectory to process.
        glob_pattern (Optional[str]): A glob pattern to filter files.
        existing_database_df (Optional[pd.DataFrame]): Existing metadata DataFrame to skip already processed files.

    Returns:
        List[str]: A list of DICOM file paths to be processed.
    """
    # Build the search pattern
    if glob_pattern:
        search_pattern = os.path.join(subdirectory, glob_pattern)
    else:
        search_pattern = os.path.join(subdirectory, '**', '*.dcm')

    dicom_files = glob.glob(search_pattern, recursive=True)
    logging.info(f"Found {len(dicom_files)} DICOM files in subdirectory: {subdirectory}")

    # Filter out already processed files
    if existing_database_df is not None and not existing_database_df.empty:
        processed_paths = set(existing_database_df['DicomStackPath'].dropna().unique())
        dicom_files = [f for f in dicom_files if os.path.dirname(f) not in processed_paths]
        logging.info(f"After filtering, {len(dicom_files)} DICOM files remain to be processed.")

    return dicom_files


def pair_dicom_series_and_rtss(
    dicom_files: List[str], desired_sop_class_uid: str
) -> Tuple[Dict[str, Dict[str, Union[str, List[str]]]], Dict[str, Dict[str, Union[str, set]]]]:
    """
    Pairs DICOM image series with RTSS files based on SeriesInstanceUID or FrameOfReferenceUID.

    Args:
        dicom_files (List[str]): List of DICOM file paths to process.
        desired_sop_class_uid (str): The SOP Class UID for image files.

    Returns:
        Tuple: Two dictionaries containing series data and RTSS data.
    """
    series_data = defaultdict(lambda: {"frame_uid": None, "images": []})
    rtss_data = {}

    # Iterate over DICOM files
    for file_path in tqdm(dicom_files, desc="Processing DICOM files", leave=False):
        logging.debug(f"Processing file: {file_path}")
        try:
            ds = pydicom.dcmread(
                file_path,
                stop_before_pixels=True,
                specific_tags=[ 
                    'SOPClassUID',
                    'FrameOfReferenceUID',
                    'SeriesInstanceUID',
                    'ReferencedFrameOfReferenceSequence',
                ],
            )
            sop_class_uid = getattr(ds, 'SOPClassUID', None)
            frame_uid = getattr(ds, 'FrameOfReferenceUID', None)
            series_uid = getattr(ds, 'SeriesInstanceUID', None)

            if sop_class_uid == desired_sop_class_uid and frame_uid and series_uid:  # Image file
                series_data[series_uid]["frame_uid"] = frame_uid
                series_data[series_uid]["images"].append(file_path)
                logging.debug(f"Added image file to series: {series_uid}, frame UID: {frame_uid}")
            elif sop_class_uid == "1.2.840.10008.5.1.4.1.1.481.3":  # RTSS file
                referenced_series_uids = set()
                if hasattr(ds, "ReferencedFrameOfReferenceSequence"):
                    for ref_frame_seq in ds.ReferencedFrameOfReferenceSequence:
                        if hasattr(ref_frame_seq, "RTReferencedStudySequence"):
                            for rt_ref_study_seq in ref_frame_seq.RTReferencedStudySequence:
                                if hasattr(rt_ref_study_seq, "RTReferencedSeriesSequence"):
                                    for rt_ref_series_seq in rt_ref_study_seq.RTReferencedSeriesSequence:
                                        series_uid = getattr(rt_ref_series_seq, "SeriesInstanceUID", None)
                                        if series_uid:
                                            referenced_series_uids.add(series_uid)
                rtss_data[file_path] = {
                    "frame_uid": frame_uid,
                    "referenced_series_uids": referenced_series_uids,
                }
                logging.debug(f"Added RTSS file: {file_path}, frame UID: {frame_uid}, referenced series: {referenced_series_uids}")
        except InvalidDicomError:
            logging.exception(f"Invalid DICOM file: {file_path}")
        except Exception as e:
            logging.exception(f"Error processing file {file_path}: {e}")

    return series_data, rtss_data


def pair_dicom_series_v5(
    folder: str,
    desired_sop_class_uid: str = "1.2.840.10008.5.1.4.1.1.4",
    glob_pattern: Optional[str] = None,
    existing_database_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Union[List[str], str]]]:
    """
    Pairs DICOM image series with RTSS files based on SeriesInstanceUID or FrameOfReferenceUID,
    with optional glob-based filtering and incremental processing.

    Args:
        folder (str): The folder containing DICOM files.
        desired_sop_class_uid (str): The SOP Class UID for image files.
        glob_pattern (Optional[str]): A glob pattern to filter files.
        convert_by_subfolders (bool): If True, process each subdirectory incrementally.
        existing_database_df (Optional[pd.DataFrame]): Existing metadata DataFrame to skip already processed files.

    Returns:
        list: A list of dictionaries containing paired image and RTSS paths.
    """
    paired_series = []

    dicom_files = filter_dicom_files(folder, glob_pattern, existing_database_df)
    series_data, rtss_data = pair_dicom_series_and_rtss(dicom_files, desired_sop_class_uid)

    # Pair Image Series with RTSS
    for series_uid, series_info in series_data.items():
        paired = False
        for rtss_path, rtss_info in rtss_data.items():
            # Try to pair Image Series with RTSS using SeriesInstanceUID references
            if series_uid in rtss_info["referenced_series_uids"]:
                images_sorted = natsorted(series_info["images"])
                paired_series.append({
                    "image": images_sorted,
                    "rtss": rtss_path
                })
                logging.info(f"Paired series {series_uid} with RTSS file: {rtss_path}")
                paired = True

            # If no specific SeriesInstanceUID is referenced, fall back to FrameOfReferenceUID
            elif (
                series_info["frame_uid"] == rtss_info["frame_uid"]
            ):
                images_sorted = natsorted(series_info["images"])
                paired_series.append({
                    "image": images_sorted,
                    "rtss": rtss_path
                })
                logging.info(f"Paired series {series_uid} with RTSS file (by FrameOfReferenceUID): {rtss_path}")
                paired = True

        # Include series without RTSS if no match found
        if not paired:
            images_sorted = natsorted(series_info["images"])
            paired_series.append({
                "image": images_sorted,
                "rtss": None
            })
            logging.info(f"Series {series_uid} has no matching RTSS file")

    return paired_series


def are_numbers_approx_equal(numbers: list, rel_tol: float) -> bool:
    """
    Determines if all numbers in a list are approximately equal to each other within a specified relative tolerance.

    Args:
        numbers (list): A list of numbers (integers or floats) to be compared.
        rel_tol (float): The relative tolerance as a non-negative number. For two numbers `a` and `b`, they are considered
          approximately equal if `|a - b| <= rel_tol * max(|a|, |b|)`.

    Returns:
        bool: True if all numbers in the list are approximately equal within the given relative tolerance; False otherwise.

    Example:
        >>> numbers = [1.00, 1.02, 0.98]
        >>> rel_tol = 0.05
        >>> are_numbers_approx_equal(numbers, rel_tol)
        True
    """

    arr = np.array(numbers)
    min_val = np.min(arr)
    return np.all(np.isclose(arr, min_val, rtol=rel_tol))

def check_approx_equal_dim1(arr, rel_tol: float) -> bool:
    return all(map(lambda x: are_numbers_approx_equal(x, rel_tol=rel_tol), arr.T))


def check_itk_image(image: sitk.Image, spacing_tolerance=1e-5) -> bool:
    """
    Check a SimpleITK image for missing slices or non-uniform sampling.

    Args:
        image: The SimpleITK image to check.

    Returns:
        True if the image is valid, False otherwise.
    """
    # Check for missing slices
    original_size = image.GetSize()
    num_slices = original_size[-1]
    expected_size = list(original_size)
    expected_size[-1] = 1
    for i in range(num_slices):
        expected_index = list(original_size)
        expected_index[-1] = i
        if not image.TransformIndexToPhysicalPoint(expected_index):
            logging.error(f"Error: missing slice {i}.")
            return False

    # Check for non-uniform sampling across slices
    spacing = image.GetSpacing()
    if (len(spacing) != 3) and (len(set(spacing)) != 1): # this checks for non-uniform sampling by first checking if there is more than one spacing value (spacing values are 3-tuples) and then checking if all spacing values are the same (set(spacing) != 1), which they should not be if you got back more than one set of spacings
        logging.warning(f"non-iniform spacing detected. Spacings = {spacing}")
        spacing_array = np.array(spacing)
        return check_approx_equal_dim1(spacing_array, 1e-6)

        # print(f"currently, there is no check for how large this discrepancy across slices is. This should be implemented, so that miniscule differences stemming from numerical noise can be disregarded.")
        # if are_numbers_approx_equal(spacing, spacing_tolerance):
        #     print(f"Spacings within tolerance of {spacing_tolerance}, continue with conversion to nifti")
        # else:
        #     # raise ValueError(f"Non-uniform spacing detected at greater than factor {1+spacing_tolerance}. Affine calculation not possible.")
        #     print(f"Non-uniform spacing at more than relative difference of {spacing_tolerance}.")
        # return False
        
    # spacing = image.GetSpacing()
    # for i in range(1, num_slices):
    #     if spacing[-1] != image.TransformIndexToPhysicalPoint([0, 0, i])[-1] - image.TransformIndexToPhysicalPoint([0, 0, i-1])[-1]:
    #         print(f"Error: non-uniform sampling between slices {i-1} and {i}.")
    #         return False

    return True

# def itk_resample_volume(
#     img: sitk.Image, 
#     new_spacing: Tuple[float, float, float],
#     interpolator = sitk.sitkLinear
# ) -> sitk.Image:
#     """
#     Resample a SimpleITK image to a new spacing using the specified interpolator.

#     Args:
#         img: The input SimpleITK image to resample.
#         new_spacing: The new spacing to resample the image to, as a tuple of (x, y, z) values.
#         interpolator: The interpolation method to use when resampling the image. Default is sitk.sitkLinear.

#     Returns:
#         The resampled SimpleITK image.
#     """
#     # Compute the new size of the resampled image
#     original_spacing = img.GetSpacing()
#     original_size = img.GetSize()
#     new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    
#     # Define the arguments for the sitk.Resample function
#     resample_args = [
#         img,                   # The input image
#         new_size,              # The new size of the output image
#         sitk.Transform(),      # The transform to apply during resampling (default is identity)
#         interpolator,          # The interpolator to use during resampling (default is sitkLinear)
#         img.GetOrigin(),       # The origin of the input image
#         new_spacing,           # The new spacing to resample the image to
#         img.GetDirection(),    # The direction of the input image
#         0,                     # The default value to use for voxels outside the input image
#         img.GetPixelID()       # The pixel ID value for the output image
#     ]
    
#     # Resample the input image
#     resampled_img = sitk.Resample(*resample_args)
    
#     return resampled_img



def itk_resample_volume(
    img: sitk.Image, 
    new_spacing: Tuple[float, float, float],
    interpolator=sitk.sitkLinear
) -> sitk.Image:
    """
    Resample a SimpleITK image to a new spacing using the specified interpolator.
    Handles both 3D and 4D images (3D images with multiple channels).

    Args:
        img: The input SimpleITK image to resample.
        new_spacing: The new spacing to resample the image to, as a tuple of (x, y, z) values.
        interpolator: The interpolation method to use when resampling the image. Default is sitk.sitkLinear.

    Returns:
        The resampled SimpleITK image.
    """
    logging.debug('Resampling nifti image using SimpleITK')
    # Get the size of the input image
    original_size = img.GetSize()
    logging.debug(f'Original image size: {original_size}')
    
    # Check if the image is 3D or 4D
    if len(original_size) == 3:
        # Resample the 3D image
        original_spacing = img.GetSpacing()
        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
        
        # Define the arguments for the sitk.Resample function
        resample_args = [
            img,                   # The input image
            new_size,              # The new size of the output image
            sitk.Transform(),      # The transform to apply during resampling (default is identity)
            interpolator,          # The interpolator to use during resampling (default is sitkLinear)
            img.GetOrigin(),       # The origin of the input image
            new_spacing,           # The new spacing to resample the image to
            img.GetDirection(),    # The direction of the input image
            0,                     # The default value to use for voxels outside the input image
            img.GetPixelID()       # The pixel ID value for the output image
        ]
        
        # Resample the input image
        resampled_img = sitk.Resample(*resample_args)
        
    elif len(original_size) == 4:
        # Get the number of channels
        num_channels = original_size[3]
        print(f"Number of channels in original image = {num_channels}")
        
        # Initialize a list to hold the resampled 3D images
        resampled_slices = []
        
        for c in range(num_channels):
            # Extract the 3D volume for the current channel
            img_3d = img[:, :, :, c]
            
            # Compute the new size for the resampled 3D volume
            original_spacing_3d = img_3d.GetSpacing()
            original_size_3d = img_3d.GetSize()
            new_size_3d = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size_3d, original_spacing_3d, new_spacing)]
            
            logging.debug(f'Original size of 3D image (channel {c}): {original_size_3d}')
            logging.debug(f'New size of 3D image (channel {c}): {new_size_3d}')
            
            # Define the arguments for the sitk.Resample function
            resample_args = [
                img_3d,                   # The input 3D image
                new_size_3d,              # The new size of the output 3D image
                sitk.Transform(),         # The transform to apply during resampling (default is identity)
                interpolator,             # The interpolator to use during resampling (default is sitkLinear)
                img_3d.GetOrigin(),       # The origin of the input image
                new_spacing,              # The new spacing to resample the image to
                img_3d.GetDirection(),    # The direction of the input image
                0,                        # The default value to use for voxels outside the input image
                img_3d.GetPixelID()       # The pixel ID value for the output image
            ]
            
            # Resample the 3D image
            resampled_img_3d = sitk.Resample(*resample_args)
            
            # Append the resampled 3D image to the list
            resampled_slices.append(resampled_img_3d)
        
        # Stack the resampled 3D images to form a 4D image
        resampled_img = sitk.JoinSeries(resampled_slices)
    
    else:
        raise ValueError("Input image must be 3D or 4D.")
    
    return resampled_img


def check_dicom_metadata_consistency(dicom_files: List[str]) -> bool:
    """
    Check the metadata consistency of a list of DICOM files across relevant attributes, ignoring any missing attributes.

    Args:
        dicom_files: A list of file paths for the DICOM files.

    Returns:
        True if the metadata is consistent across relevant attributes, False otherwise.
    """
    # Read the first DICOM file to obtain the metadata attributes of interest
    first_dicom_data = pydicom.dcmread(dicom_files[0])
    attributes_of_interest = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'Modality']
    metadata_dict = {attr: getattr(first_dicom_data, attr, '') for attr in attributes_of_interest}

    # Check the metadata consistency across relevant attributes for all DICOM files
    for file_path in dicom_files[1:]:
        dicom_data = pydicom.dcmread(file_path)
        for attr in attributes_of_interest:
            if getattr(dicom_data, attr, '') != '' and getattr(dicom_data, attr, '') != metadata_dict[attr]:
                return False

    return True

def convert_photometric_interpretation(image: sitk.Image, target_interpretation: str) -> sitk.Image:
    """
    Convert the photometric interpretation of a SimpleITK image between MONOCHROME1 and MONOCHROME2.

    This function reads the image's metadata to determine its current photometric interpretation.
    If the current and target interpretations are different, it inverts the grayscale intensities
    and updates the metadata accordingly.

    Args:
        image (sitk.Image): The input SimpleITK image with MONOCHROME1 or MONOCHROME2 photometric interpretation.
        target_interpretation (str): The desired photometric interpretation ('MONOCHROME1' or 'MONOCHROME2').

    Returns:
        sitk.Image: The converted SimpleITK image with the desired photometric interpretation.

    Raises:
        RuntimeError: If the input image does not have the '0028|0004' (Photometric Interpretation) metadata key.
        ValueError: If the current or target photometric interpretation is not supported (i.e., not 'MONOCHROME1' or 'MONOCHROME2').
    """
    if not image.HasMetaDataKey('0028|0004'):
        raise RuntimeError("Input image does not have the '0028|0004' (Photometric Interpretation) metadata key")

    current_interpretation = image.GetMetaData('0028|0004').strip()

    if current_interpretation not in ['MONOCHROME1', 'MONOCHROME2']:
        raise ValueError(f"Unsupported photometric interpretation '{current_interpretation}'")

    if target_interpretation not in ['MONOCHROME1', 'MONOCHROME2']:
        raise ValueError(f"Unsupported target photometric interpretation '{target_interpretation}'")

    if current_interpretation != target_interpretation:
        logging.info(f"converting from '{current_interpretation}' to '{target_interpretation}'")
        image_arr = sitk.GetArrayFromImage(image)
        inverted_arr = np.max(image_arr) - image_arr
        image = sitk.GetImageFromArray(inverted_arr)
        image.SetMetaData('0028|0004', target_interpretation)
        
    return image



def load_dicom_images(dicom_path: Union[str, List[str]], new_spacing: Optional[Tuple[float, float, float]] = None, 
                      orientation: Optional[str] = None, permute_axes: Optional[List[int]] = None, 
                      photometric_interpretation: Optional[str] = 'MONOCHROME2') -> sitk.Image:
    """
    Load a series of DICOM images into a SimpleITK image object from a list of DICOM file paths,
    a path to a single DICOM file, or the path to a directory containing DICOM files from a single
    series.

    The function checks for consistency of metadata in the DICOM files. Also excludes RTStruct files.
    
    In case of inconsistent slice widths, will automatically resample voxel spacing to (1,1,1) mm.

    Args:
        dicom_path (Union[str, List[str]]): A list of file paths for the DICOM files, a single path, or a directory.
        new_spacing (Optional[Tuple[float, float, float]]): The desired voxel spacing of the output image. 
                                                           If provided, the image will be resampled to this resolution.
                                                           Default is None, which means the image will not be resampled.
        orientation (Optional[str]): A string specifying the desired axes orientation 
                                           of the output image. Default is None, which means the original orientation 
                                           will be preserved.
        permute_axes (Optional[List[int]]): A list of three integers that specifies the desired order of axes in the 
                                            output image.
        photometric_interpretation (Optional[str]): The desired photometric interpretation for the output image. 
                                                    Supported values: 'MONOCHROME1' or 'MONOCHROME2'.
                                                    Default is None, which means the original photometric interpretation
                                                    will be preserved.

    Returns:
        A SimpleITK image object containing the loaded DICOM images.

    Raises:
        RuntimeError: If no files were found, if the input path is not valid or if the series could not be read.
        ValueError: If the DICOM metadata is not consistent across all files, or if the input arguments are not valid.
    """             
    if os.path.isdir(dicom_path):
        dicom_files = get_dicom_files(dicom_path)
    else:
        if isinstance(dicom_path, str):
            dicom_files = [dicom_path]
        else:
            dicom_files = dicom_path

    if not check_dicom_metadata_consistency(dicom_files):
        raise RuntimeError("Inconsistent DICOM files! 'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', or 'Modality' are not the same across all files")
    
    # Check if the input DICOM file list exists
    if not dicom_files:
        raise RuntimeError("Empty DICOM file list.")
    
    # Read metadata from the first DICOM file
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(dicom_files[0])
    file_reader.ReadImageInformation()

    metadata_keys = file_reader.GetMetaDataKeys()
    metadata_dict = {key: file_reader.GetMetaData(key) for key in metadata_keys}

    # Load DICOM series
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    # Set metadata in the output image
    for key, value in metadata_dict.items():
        image.SetMetaData(key, value)
    
    # Convert photometric interpretation if specified
    if photometric_interpretation is not None:
        image = convert_photometric_interpretation(image, photometric_interpretation)
    
    if new_spacing is not None:
        logging.info(f'resampling image to resolution {new_spacing} mm')
        image = itk_resample_volume(img=image, new_spacing=new_spacing)
    
    if not check_itk_image(image):
        raise RuntimeError("Inconsistent slice widths detected! You should resample the dicom images to a uniform voxel spacing. Make sure to also resample any rtss images")
        # print('resampling image due to inconsistencies in original image')
        # print('setting new spacing to (1,1,1) mm')
        # image = itk_resample_volume(img=image, new_spacing=(1,1,1))
    
    if permute_axes is not None:
        image = permute_itk_axes(image, permute_axes=permute_axes)
    
    if orientation is not None:
        image = sitk.DICOMOrient(image, orientation)
    
    return image

def get_dicom_files(dicom_folder: str) -> List[str]:
    """
    Find all DICOM files in a directory and filter out RTSTRUCT files.

    Args:
        dicom_folder: The path to the folder containing the DICOM files.

    Returns:
        A list of file paths for the DICOM files that do not have a Modality attribute or have a Modality
        attribute that is not 'RTSTRUCT'.
    """
    dicom_files = [os.path.join(dicom_folder, entry.name) for entry in os.scandir(dicom_folder) if entry.name.endswith('.dcm')]
    
    dicom_files = [f for f in dicom_files if 'RTSTRUCT' not in pydicom.dcmread(f, stop_before_pixels=True).get('Modality', '')]
    
    # Sort the DICOM files only if there are 3D images
    first_slice = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
    if 'ImagePositionPatient' in first_slice:
        dicom_files.sort(key=lambda x: pydicom.dcmread(x, stop_before_pixels=True).get('ImagePositionPatient', '')[2])
        
    return dicom_files

def get_affine_from_itk_image(image: sitk.Image, spacing_tolerance=1e-5) -> np.ndarray:
    """
    Calculate the affine transformation matrix for a SimpleITK image based on the metadata of the DICOM files.

    Args:
        image: The SimpleITK image to calculate the affine transformation for.

    Returns:
        The affine transformation matrix as a NumPy array.

    Raises:
        ValueError: If the SimpleITK image is not 2D or 3D or if the image spacing is not uniform.
    """
    dimension = image.GetDimension()
    if dimension not in [2, 3]:
        raise ValueError("Invalid image dimension. Only 2D and 3D images are supported.")
    
    spacing = image.GetSpacing()
    if len(set(spacing)) != 1:
        # print(f"non-iniform spacing detected. Spacings = {spacing}")
        if not are_numbers_approx_equal(spacing, spacing_tolerance):
            # print(f"Spacings within tolerance of {spacing_tolerance}, continue with conversion to nifti")
        # else:
            raise ValueError(f"Non-uniform spacing ({spacing}) with more than relative difference of {spacing_tolerance} detected. Affine calculation not possible.")


    matrix = np.array(image.GetDirection()).reshape((dimension, dimension))
    origin = np.array(image.GetOrigin())

    if dimension == 3 and matrix[2, 2] == 0:
        # The DICOM files have the slice order flipped, so flip the sign of the z-component of the matrix
        matrix[2, :] *= -1
        origin[2] += (image.GetSize()[2] - 1) * spacing[-1]

    affine = np.zeros((dimension + 1, dimension + 1))
    affine[:-1, :-1] = matrix
    affine[:-1, -1] = origin
    affine[-1, -1] = 1

    return affine

# def save_itk_image_as_nifti_nib(image: sitk.Image, file_path: str) -> None:
#     """
#     DEPRECATED, USE save_itk_image_as_nifti_sitk INSTEAD
# 
#     Saves a SimpleITK image as a Nifti file using nibabel.

#     Args:
#         image (sitk.Image): The SimpleITK image to save.
#         file_path (str): The path to the output Nifti file.

#     Raises:
#         ValueError: If the image has a non-supported pixel type.
#     """
#     # Convert the SimpleITK image to a Numpy array
#     np_array = sitk.GetArrayFromImage(image)

#     # Get the affine transformation 
#     affine = get_affine_from_itk_image(image)

#     # Create a Nifti1Image from the Numpy array and affine
#     nifti_image = nib.Nifti1Image(np_array, affine)

#     # Get some metadata from the original SimpleITK image
#     direction = np.array(image.GetDirection()).reshape((3, 3))
#     origin = np.array(image.GetOrigin())
#     spacing = np.array(image.GetSpacing())
#     dimensions = np.array(image.GetSize())
    
#     # Add the metadata to the NIfTI header
#     qform = np.eye(4)
#     qform[0:3, 0:3] = np.array(direction).reshape((3, 3)) * np.array(spacing).reshape((3, 1))
#     qform[0:3, 3] = origin
    

#     sform = np.eye(4)
#     sform[0:3, 0:3] = np.array(direction).reshape((3, 3)) * np.array(spacing).reshape((3, 1))
#     sform[0:3, 3] = origin
    
    
#     # Add the metadata to the NIfTI header
#     # nifti_image.header['q_form'](qform, code=1)
#     # nifti_image.header['s_form'](sform, code=1)
#     nifti_image.header.set_qform(qform)
#     nifti_image.header.set_sform(sform)
#     nifti_image.header['qoffset_x'] = origin[0]
#     nifti_image.header['qoffset_y'] = origin[1]
#     nifti_image.header['qoffset_z'] = origin[2]
#     nifti_image.header['pixdim'][1:4] = spacing
#     nifti_image.header['dim'][1:4] = dimensions

#     # Save the Nifti1Image to file
#     try:
#         nib.save(nifti_image, file_path)
#     except Exception as e:
#         raise ValueError(f"Error saving image to file: {e}")

def save_itk_image_as_nifti_sitk(image: sitk.Image, file_path: str) -> None:
    """
    Saves a SimpleITK image as a Nifti file.

    Args:
        image (sitk.Image): The SimpleITK image to save.
        file_path (str): The path to the output Nifti file.

    Raises:
        RuntimeError: If the image has a non-supported pixel type or if there is an error saving the Nifti file.
    """
    try:
        sitk.WriteImage(image, file_path)
    except RuntimeError as e:
        raise RuntimeError(f"Error saving image to file: {e}")    


def permute_itk_axes(image, permute_axes=[1,0,2]) -> sitk.Image:
    """
    Permutes the axes of a SimpleITK image according to the specified order.

    This function creates a filter to permute the axes of an image, which can be useful for reorienting images
    before processing or saving them.

    Args:
        image (sitk.Image): The SimpleITK image to permute.
        permute_axes (List[int], optional): The desired order of the axes after permutation. 
                                             Default is [1, 0, 2], which swaps the first and second axes.

    Returns:
        sitk.Image: The image after axes permutation.

    Example:
        ```python
        image = sitk.ReadImage('path/to/image')
        permuted_image = permute_itk_axes(image, [2, 1, 0])
        ```
    """

    # create a filter to permute axes
    permute_filter = sitk.PermuteAxesImageFilter()
    permute_filter.SetOrder(permute_axes)  # permute the axes
    # apply the filter to the image
    image = permute_filter.Execute(image)
    return image