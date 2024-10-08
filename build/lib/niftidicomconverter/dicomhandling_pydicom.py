import os
import logging 

import numpy as np
import SimpleITK as sitk
import nibabel as nib

import pydicom
import scipy.ndimage

from typing import Tuple, Optional, List, Union
from pydicom.encaps import encapsulate
from scipy.signal import resample_poly
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from .dicomhandling import check_dicom_metadata_consistency

##########################################################################################
##########################################################################################
# below this point we have similar functions, but using pydicom instead of sitk
##########################################################################################
##########################################################################################

def check_dicom_array_consistency_pydicom(slices: List[pydicom.dataset.FileDataset]) -> bool:
    """
    Checks the consistency of a list of DICOM slices.

    Parameters:
    slices (List[pydicom.dataset.FileDataset]): A list of DICOM slices.

    Returns:
    bool: True if the slices are consistent, False otherwise.
    """

    if not (hasattr(slices[0], 'SliceThickness') | hasattr(slices[0], 'ImageOrientationPatient') | hasattr(slices[0], 'SliceLocation')):
        logging.warning("The relevant attributes for which to check consistency are missing from the file")
        return True

    # Check for missing slices
    num_slices = len(slices)
    num_images = slices[-1].InstanceNumber - slices[0].InstanceNumber + 1
    if num_slices != num_images:
        logging.warning(f"Missing slices: expected {num_images}, got {num_slices}.")
        return False

    # Check for consistent slice thickness
    
    slice_thicknesses = [s.SliceThickness for s in slices]
    if len(set(slice_thicknesses)) > 1:
        logging.warning("Inconsistent slice thickness.")
        return False

    # Check for consistent orientations
    orientations = [(*s.ImageOrientationPatient, s.SliceLocation) for s in slices]
    if len(set(orientations)) > 1:
        logging.warning("Inconsistent orientations.")
        return False

    logging.warning("DICOM slices are consistent.")
    return True

def resample_volume_pydicom(image: np.ndarray, spacing: np.ndarray, new_spacing: Tuple[float, float, float], order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a 3D image to a new voxel spacing.

    Args:
        image (np.ndarray): A 3D numpy array representing the input image.
        spacing (np.ndarray): A 3-element numpy array representing the current voxel spacing of the input image.
        new_spacing (Tuple[float, float, float]): A 3-element tuple representing the desired voxel spacing of the output image.
        order (int): The order of interpolation used to resample the image. Default is 3.

    Returns:
        A tuple containing the resampled 3D image and its new voxel spacing.
    """
    resize_factor = np.array(spacing) / np.array(new_spacing)
    # new_shape = np.round(image.shape * resize_factor)
    new_spacing = np.array(spacing) / np.array(resize_factor)
    image = scipy.ndimage.interpolation.zoom(image, resize_factor, order=order)
    return image, new_spacing



def orient_volume_pydicom(image: np.ndarray, orientation: str) -> np.ndarray:
    """
    Orient a 3D image according to a desired axes orientation.

    Args:
        image (np.ndarray): A 3D numpy array representing the input image.
        orientation (str): A string representing the desired axes orientation of the output image.
                           Valid values are "RAS", "LPS", "RAI", "LPI", "ARS", "PRS", "AIP", "PIP".

    Returns:
        A 3D numpy array representing the oriented image.
    """
    directions = {
        'RAS': ((0, 1, 2), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        'LPS': ((0, 1, 2), (-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        'RAI': ((0, 1, 2), (1, 0, 0), (0, 0, -1), (0, -1, 0)),
        'LPI': ((0, 1, 2), (-1, 0, 0), (0, 0, -1), (0, 1, 0)),
        'ARS': ((1, 0, 2), (0, 1, 0), (1, 0, 0), (0, 0, 1)),
        'PRS': ((1, 0, 2), (0, -1, 0), (1, 0, 0), (0, 0, 1)),
        'AIP': ((1, 0, 2), (0, 1, 0), (0, 0, 1), (0, 0, -1)),
        'PIP': ((1, 0, 2), (0, -1, 0), (0, 0, -1), (0, 1, 0))
    }

    if orientation not in directions:
        raise ValueError(f"Invalid orientation '{orientation}'. Valid orientations are {list(directions.keys())}.")

    permute_axes, transpose_matrix, flip_matrix, _ = directions[orientation]
    image = np.transpose(image, permute_axes)
    image = np.matmul(transpose_matrix, image)
    image = np.flip(image, flip_matrix)
    return image

def convert_photometric_interpretation_pydicom(dcm: pydicom.dataset.FileDataset, target_interpretation: str) -> pydicom.dataset.FileDataset:
    """
    Convert the photometric interpretation of a Pydicom dataset to the specified target interpretation.

    Args:
        dcm (pydicom.dataset.FileDataset): The input Pydicom dataset.
        target_interpretation (str): The target photometric interpretation, either 'MONOCHROME1' or 'MONOCHROME2'.

    Returns:
        A Pydicom dataset.FileDataset with the updated photometric interpretation and pixel data.

    Raises:
        ValueError: If the target photometric interpretation is not supported.
    """
    current_interpretation = dcm.PhotometricInterpretation

    if current_interpretation not in ['MONOCHROME1', 'MONOCHROME2']:
        raise ValueError(f"Unsupported photometric interpretation '{current_interpretation}'")

    if target_interpretation not in ['MONOCHROME1', 'MONOCHROME2']:
        raise ValueError(f"Unsupported target photometric interpretation '{target_interpretation}'")

    if current_interpretation != target_interpretation:
        modified_pixel_array = (np.iinfo(dcm.pixel_array.dtype).max - dcm.pixel_array).astype(dcm.pixel_array.dtype)
        dcm.PixelData = encapsulate([modified_pixel_array.tostring()])
        dcm.PhotometricInterpretation = target_interpretation

    return dcm




def load_dicom_images_pydicom(paths: Union[str, List[str]], new_spacing: Optional[Tuple[float, float, float]] = None, 
                      orientation: Optional[str] = None, permute_axes: Optional[List[int]] = None, 
                      photometric_interpretation: Optional[str] = None, **kwargs) -> pydicom.dataset.FileDataset:
    """
    Loads DICOM image files into a single DICOM image dataset using the Pydicom library.

    Parameters:
    paths (Union[str, List[str]]): The path(s) to the DICOM file(s).
    new_spacing (Optional[Tuple[float, float, float]]): The new voxel spacing in mm for the resampled image. 
    If None, the image is not resampled. Defaults to None.
    orientation (Optional[str]): The new orientation of the image. If None, the orientation is not changed. 
    Defaults to None.
    permute_axes (Optional[List[int]]): The order in which to permute the axes of the image. If None, 
    the axes are not permuted. Defaults to None.

    Returns:
    pydicom.dataset.FileDataset: A single DICOM image dataset.

    Raises:
    RuntimeError: If the DICOM files are not consistent, i.e., 'PatientID', 'StudyInstanceUID', 
    'SeriesInstanceUID', or 'Modality' are not the same across all files, or if there is an error 
    in one or more DICOM files.

    """

    if isinstance(paths, str):
        if os.path.isdir(paths):
            paths = [os.path.join(paths, f) for f in os.listdir(paths) if 'RTSTRUCT' not in pydicom.dcmread(os.path.join(paths, f), stop_before_pixels=True).get('Modality', '')]
        else:
            paths = [paths]
            
    if not check_dicom_metadata_consistency(paths):
        raise RuntimeError("Inconsistent DICOM files! 'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', or 'Modality' are not the same across all files")

    # Sort files by instance number
    slices = []
    for path in paths:
        dcm = pydicom.dcmread(path, **kwargs)
        slices.append(dcm)

    slices.sort(key=lambda x: x.InstanceNumber)
    
    # Combine images into a single DICOM series
    if len(slices) > 1:
        pixel_arrays = [dcm.pixel_array for dcm in slices]
        combined_pixel_array = np.dstack(pixel_arrays)
        image = slices[0]
        image.PixelData = combined_pixel_array.tostring()
        image.NumberOfFrames = combined_pixel_array.shape[2]
    else:
        image = slices[0]
    
    # Convert photometric interpretation if specified
    if photometric_interpretation is not None:
        image = convert_photometric_interpretation_pydicom(image, photometric_interpretation)
    
    # calculate voxel spacing
    if len(slices[0].pixel_array.shape) == 3:
        spacing = [(image.PixelSpacing[0], image.PixelSpacing[1], image.SliceThickness) for image in slices]
    else:
        # spacing = [(image.PixelSpacing[0], image.PixelSpacing[1], 1.0) for image in slices]
        if hasattr(slices[0], 'PixelSpacing'):
            spacing = (image.PixelSpacing[0], image.PixelSpacing[1], 1.0)
        else:
            spacing = (1.0, 1.0, 1.0)
    
    # set new_spacing to (1,1,1) in case the slice thicknesses are different, and there is no preset new_spacing
    if ((len(set(spacing)) > 1) or (not check_dicom_array_consistency_pydicom(slices))) and (new_spacing is None):
        # new_spacing = [(1,1,1) for _ in slices]
        new_spacing = (1,1,1)
        logging.warning('resampling image due to inconsistent slice thicknesses in original image')
        logging.warning('setting new spacing to (1,1,1) mm')
    
    # Resample voxel spacing
    if new_spacing is not None:
        logging.info(f'resampling image to resolution {new_spacing} mm')
        if len(set(spacing)) == 1:
            spacing = tuple([float(s) for s in list(set(spacing))[0]])
        else:
            new_spacing = [new_spacing for _ in slices]
            
        pixel_array, spacing = resample_volume_pydicom(image.pixel_array, spacing, new_spacing)
        if len(slices) > 1:
            image.PixelData = pixel_array.tostring()
            image.Rows, image.Columns, image.NumberOfFrames = pixel_array.shape
        else:
            image.PixelData = pixel_array[:,:,0].tostring()
            image.Rows, image.Columns = pixel_array[:,:,0].shape
    
    if not check_dicom_array_consistency_pydicom(slices):
        # raise RuntimeError("Error in dicom file(s)!")
        logging.error(f'note error in dicom file(s)\n{paths}')

    # Transpose axes if necessary
    if permute_axes is not None:
        pixel_array = np.transpose(image.pixel_array, permute_axes)
        image.PixelData = pixel_array.tostring()

    # Orient image if necessary
    if orientation is not None:
        image.pixel_array = orient_volume_pydicom(image.pixel_array, orientation)
        image.PixelData = pixel_array.tostring()

    return image

def save_pydicom_to_nifti_nib(dicom_data: pydicom.dataset.FileDataset, nifti_file_path: str) -> Tuple[bool, str]:
    """
    Convert a PyDICOM dataset to a NIfTI file using NiBabel.

    :param dicom_data: The PyDICOM dataset to convert.
    :type dicom_data: pydicom.dataset.FileDataset
    :param nifti_file_path: The desired path for the NIfTI file.
    :type nifti_file_path: str
    :return: A tuple indicating whether the conversion was successful (True or False) and a message describing the result.
    :rtype: Tuple[bool, str]
    """
    try:
        if hasattr(dicom_data, 'affine'):
            affine = dicom_data.affine
        else:
            try:
                affine = calculate_affine_from_pydicom(dicom_data)
            except:
                logging.warning('cannot calculate affine from dicom, set it to identity matrix')
                affine = np.ones((4,4))
        nifti_data = nib.nifti1.Nifti1Image(dicom_data.pixel_array, affine)
        nib.save(nifti_data, nifti_file_path)
        logging.info(f"Conversion successful. NIfTI file saved at {nifti_file_path}")
        return True
    except Exception as e:
        error_message = f"Error converting DICOM to NIfTI: {str(e)}"
        logging.exception(error_message)
        return False
    
def calculate_affine_from_pydicom(dicom_image: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Calculate the affine matrix for a PyDICOM image.

    :param dicom_image: The PyDICOM image for which to calculate the affine matrix.
    :type dicom_image: pydicom.dataset.FileDataset
    :return: The affine matrix for the PyDICOM image.
    :rtype: numpy.ndarray
    """
    # Extract the necessary metadata from the DICOM image
    spacing = np.array(dicom_image.PixelSpacing + [dicom_image.SliceThickness], dtype=float)
    orientation = np.array(dicom_image.ImageOrientationPatient, dtype=float).reshape(2, 3)
    origin = np.array(dicom_image.ImagePositionPatient + [1], dtype=float)

    # Calculate the affine matrix
    affine = np.zeros((4, 4))
    affine[:3, :3] = np.matmul(orientation, np.diag(spacing))
    affine[:3, 3] = origin[:3]
    affine[3, 3] = 1.0

    return affine

def pydicom_to_simpleitk(dcm: pydicom.dataset.FileDataset) -> sitk.Image:
    """
    Convert a Pydicom dataset to a SimpleITK image object.

    Args:
        dcm (pydicom.dataset.FileDataset): The input Pydicom dataset.

    Returns:
        A SimpleITK image object with the same pixel data and metadata as the input Pydicom dataset.
    """
    pixel_array = dcm.pixel_array.astype(np.float32)
    image = sitk.GetImageFromArray(pixel_array)

    for key, value in dcm.items():
        if hasattr(sitk, key):
            setattr(image, key, value)
        else:
            image.SetMetaData(key, str(value))

    return image

################################################################################
# upsampling contours #
################################################################################

def smooth_and_upsample_contours(contours, window_length, polyorder, upsample_factor, kind='quadratic'):
    """
    Smooths and upsamples the given contours using Savitzky-Golay smoothing and linear interpolation.

    Args:
        contours (list of numpy.ndarray): List of contours to be smoothed and upsampled. Each contour is a 2D array where each row represents a point (x, y, z).
        window_length (int): The length of the window for the Savitzky-Golay filter. Must be an odd integer.
        polyorder (int): The order of the polynomial used in the Savitzky-Golay filter.
        upsample_factor (int): The factor by which to upsample the contours.
        kind (str): The kind of interpolation used for upsampling. Default is 'quadratic'.

    Returns:
        list of list: The smoothed and upsampled contours. Each contour is flattened into a 1D list.

    Raises:
        ValueError: If `window_length` is not a positive odd integer.
        ValueError: If `polyorder` is not a non-negative integer less than `window_length`.
        ValueError: If `upsample_factor` is not a positive integer.
    """
    smoothed_and_upsampled_contours = []
    

    for i, contour in enumerate(contours):
        logging.info(f"Smoothing and upsampling {len(contour)} contours...")
        # Reshape the contour data
        contour = np.reshape(contour, (-1, 3))

        # Create a new array for the upsampled contour
        upsampled_contour = np.zeros((contour.shape[0] * upsample_factor, 3))

        # Upsample the x and y coordinates of the contour data
        # upsampled_contour[:, :2] = resample_poly(contour[:, :2], upsample_factor, 1)

        # Create an array of the original indices
        old_indices = np.arange(contour.shape[0])

        # Create an array of the upsampled indices
        new_indices = np.linspace(0, contour.shape[0] - 1, contour.shape[0] * upsample_factor)

        # Perform linear interpolation for the x and y coordinates
        for j in range(2):
            interpolator = interp1d(old_indices, contour[:, j], kind=kind)
            upsampled_contour[:, j] = interpolator(new_indices)

        # Copy the z coordinates from the original contour
        upsampled_contour[:, 2] = np.repeat(contour[:, 2], upsample_factor)

        # Check if the contour is too small
        if upsampled_contour.shape[0] < window_length:
            # If so, notify the user and skip the smoothing operation
            logging.warning(f"Contour {i} has fewer points ({upsampled_contour.shape[0]}) than the window length ({window_length}). Skipping smoothing.")
            smoothed_and_upsampled_contours.append(upsampled_contour.flatten().tolist())
            continue

        # Apply Savitzky-Golay smoothing
        upsampled_contour = savgol_filter(upsampled_contour, window_length, polyorder, axis=0)
        # smoothed_contour = upsampled_contour

        # Flatten the smoothed contour data and convert it to a list
        upsampled_contour = upsampled_contour.flatten().tolist()

        smoothed_and_upsampled_contours.append(upsampled_contour)
        logging.info(f'len(smoothed_contour) = {len(upsampled_contour)}')
        
    return smoothed_and_upsampled_contours

def update_contours_in_dicom_file(dicom_file_path, window_length, polyorder, upsample_factor, upsampling_kind='quadratic'):
    """
    Update the contours in a DICOM file by smoothing and upsampling the contour data.

    The upsampling happens first and the smoothing second. 

    Args:
        dicom_file_path (str): The path to the DICOM file.
        window_length (int): The length of the smoothing window.
        polyorder (int): The order of the polynomial used for smoothing.
        upsample_factor (float): The factor by which to upsample the contour data.
        upsampling_kind (str): The kind of interpolation used for upsampling. Default is 'quadratic'.

    Returns:
        None
    """
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_file_path)

    # Get the ROI Contour Sequence
    roi_contour_sequence = ds.ROIContourSequence

    # Iterate over each ROI Contour Sequence item
    for roi_contour in roi_contour_sequence:
        # Get the Contour Sequence
        contour_sequence = roi_contour.ContourSequence

        # Iterate over each Contour Sequence item
        for contour_index, contour in enumerate(contour_sequence):
            # Get the contour data
            contour_data = contour.ContourData

            # Smooth and upsample the contour data
            smoothed_and_upsampled_contour_data = smooth_and_upsample_contours([contour_data], window_length, polyorder, upsample_factor, kind=upsampling_kind)

            # Update the contour data
            contour.ContourData = smoothed_and_upsampled_contour_data[0]

            # Update the number of contour points
            contour.NumberOfContourPoints = len(smoothed_and_upsampled_contour_data[0]) // 3

    # Save the modified DICOM file
    ds.save_as(dicom_file_path)

def read_dicom_stack(directory):
    """
    Read DICOM files from the specified directory and return a list of DICOM objects.

    Args:
        directory (str): The path to the directory containing the DICOM files.

    Returns:
        list: A list of pydicom objects representing the DICOM files in the directory.
    """
    dicom_stack = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.dcm'):
            dicom_stack.append(pydicom.dcmread(filepath))
    return dicom_stack