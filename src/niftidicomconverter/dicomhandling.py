import SimpleITK as sitk
from typing import List, Union, Optional
import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional
import nibabel as nib

def check_itk_image(image: sitk.Image) -> bool:
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
            print(f"Error: missing slice {i}.")
            return False

    # Check for non-uniform sampling
    spacing = image.GetSpacing()
    for i in range(1, num_slices):
        if spacing[-1] != image.TransformIndexToPhysicalPoint([0, 0, i])[-1] - image.TransformIndexToPhysicalPoint([0, 0, i-1])[-1]:
            print(f"Error: non-uniform sampling between slices {i-1} and {i}.")
            return False

    return True

def itk_resample_volume(
    img: sitk.Image, 
    new_spacing: Tuple[float, float, float],
    interpolator = sitk.sitkLinear
) -> sitk.Image:
    """
    Resample a SimpleITK image to a new spacing using the specified interpolator.

    Args:
        img: The input SimpleITK image to resample.
        new_spacing: The new spacing to resample the image to, as a tuple of (x, y, z) values.
        interpolator: The interpolation method to use when resampling the image. Default is sitk.sitkLinear.

    Returns:
        The resampled SimpleITK image.
    """
    # Compute the new size of the resampled image
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
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

def load_dicom_images(dicom_path: Union[str, List[str]], new_spacing: Optional[Tuple[float, float, float]] = None, 
                      orientation: Optional[str] = None, permute_axes: Optional[List[int]] = None) -> sitk.Image:
    """
    Load a series of DICOM images into a SimpleITK image object from a list of DICOM file paths,
    a path to a single DICOM file, or the path to a directory containing DICOM files from a single
    series.

    The function checks for consistency of metadata in the DICOM files. Also exluces RTStruct files.
    
    In case of inconsistent slice widths, will automatically resample voxel spacing to (1,1,1) mm.


    Args:
        dicom_path (Union[str, List[str]]): A list of file paths for the DICOM files, a single path, or a directory.
        new_spacing (Optional[Tuple[float, float, float]]): The desired voxel spacing of the output image. 
                                                           If provided, the image will be resampled to this resolution.
                                                           Default is None, which means the image will not be resampled.
        orientation (Optional[List[int]]): A list of three integers that specifies the desired axes orientation 
                                           of the output image. Default is None, which means the original orientation 
                                           will be preserved.
        permute_axes (Optional[List[int]]): A list of three integers that specifies the desired order of axes in the 
                                            output image.
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

    # Load DICOM series
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    
    if new_spacing is not None:
        print(f'resampling image to resolution {new_spacing} mm')
        image = itk_resample_volume(img=image, new_spacing=new_spacing)
    
    if not check_itk_image(image):
        print('resampling image due to inconsistencies in original image')
        print('setting new spacing to (1,1,1) mm')
        image = itk_resample_volume(img=image, new_spacing=(1,1,1))
    
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

def get_affine_from_itk_image(image: sitk.Image) -> np.ndarray:
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
        raise ValueError("Non-uniform spacing detected. Affine calculation is not possible.")

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

def save_itk_image_as_nifti_nibabel(image: sitk.Image, file_path: str) -> None:
    """
    Saves a SimpleITK image as a Nifti file using nibabel.

    Args:
        image (sitk.Image): The SimpleITK image to save.
        file_path (str): The path to the output Nifti file.

    Raises:
        ValueError: If the image has a non-supported pixel type.
    """
    # Convert the SimpleITK image to a Numpy array
    np_array = sitk.GetArrayFromImage(image)

    # Get the affine transformation 
    affine = get_affine_from_itk_image(image)

    # Create a Nifti1Image from the Numpy array and affine
    nifti_image = nib.Nifti1Image(np_array, affine)

    # Get some metadata from the original SimpleITK image
    direction = np.array(image.GetDirection()).reshape((3, 3))
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    dimensions = np.array(image.GetSize())
    
    # Add the metadata to the NIfTI header
    qform = np.eye(4)
    qform[0:3, 0:3] = np.array(direction).reshape((3, 3)) * np.array(spacing).reshape((3, 1))
    qform[0:3, 3] = origin
    

    sform = np.eye(4)
    sform[0:3, 0:3] = np.array(direction).reshape((3, 3)) * np.array(spacing).reshape((3, 1))
    sform[0:3, 3] = origin
    
    
    # Add the metadata to the NIfTI header
    # nifti_image.header['q_form'](qform, code=1)
    # nifti_image.header['s_form'](sform, code=1)
    nifti_image.header.set_qform(qform)
    nifti_image.header.set_sform(sform)
    nifti_image.header['qoffset_x'] = origin[0]
    nifti_image.header['qoffset_y'] = origin[1]
    nifti_image.header['qoffset_z'] = origin[2]
    nifti_image.header['pixdim'][1:4] = spacing
    nifti_image.header['dim'][1:4] = dimensions

    # Save the Nifti1Image to file
    try:
        nib.save(nifti_image, file_path)
    except Exception as e:
        raise ValueError(f"Error saving image to file: {e}")

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

    # # Try to copy as much metadata as possible
    # try:
    #     itk_origin = image.GetOrigin()
    #     itk_spacing = image.GetSpacing()
    #     itk_direction = image.GetDirection()
    #     itk_size = image.GetSize()

    #     itk_meta_data = {
    #         "origin": [float(o) for o in itk_origin],
    #         "spacing": [float(s) for s in itk_spacing],
    #         "direction": [float(d) for d in itk_direction],
    #         "size": [int(s) for s in itk_size],
    #     }

    #     for key in image.GetMetaDataKeys():
    #         value = image.GetMetaData(key)
    #         try:
    #             itk_meta_data[key] = float(value)
    #         except ValueError:
    #             try:
    #                 itk_meta_data[key] = int(value)
    #             except ValueError:
    #                 itk_meta_data[key] = value
    #     print(f'Image meta data = {itk_meta_data}')
        
    #     # sitk.WriteImage(sitk.ReadImage(file_path), file_path, True, itk_meta_data)
    #     output_image = sitk.ReadImage(file_path)
    #     output_image.CopyInformation(image)
    #     sitk.WriteImage(output_image, file_path, True, itk_meta_data)
        
    # except Exception as e:
    #     print(f"Warning: Could not copy all metadata to output file: {e}")
        
    


def permute_itk_axes(image, permute_axes=[1,0,2]):
    # create a filter to permute axes
    permute_filter = sitk.PermuteAxesImageFilter()
    permute_filter.SetOrder(permute_axes)  # permute the axes
    # apply the filter to the image
    image = permute_filter.Execute(image)
    return image