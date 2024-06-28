import numpy as np
import nibabel as nib
import tempfile
import os
import logging

from typing import List, Union, Dict, Optional
from rt_utils import RTStructBuilder

import dicom2nifti

from niftidicomconverter.dicomhandling import load_dicom_images, save_itk_image_as_nifti_sitk
from niftidicomconverter.dicomhandling_pydicom import load_dicom_images_pydicom, save_pydicom_to_nifti_nib
from niftidicomconverter.utils import copy_file_safely
from niftidicomconverter.rois import fetch_rtstruct_roi_masks

def convert_dicom_series_to_nifti_image(dicom_image_files, output_fname):
    """
    Converts a series of DICOM images to a NIfTI image.
    
    Uses a different backend than the `convert_dicom_to_nifti` function, which uses SimpleITK or pydicom.

    Args:
        dicom_image_files (list or str): A list of DICOM image file paths or a single DICOM image file path.
        output_fname (str): The output file name for the NIfTI image.

    Returns:
        None

    Raises:
        None
    """

    if not isinstance(dicom_image_files, list):
        dicom_image_files = [dicom_image_files]

    temp_dir = tempfile.mkdtemp()
    # copy all image files into the temp directory
    for image_file in dicom_image_files:
        copy_file_safely(image_file, temp_dir)
        
    nii_result = dicom2nifti.dicom_series_to_nifti(
        temp_dir,
        output_fname,
        reorient_nifti=False,
    )
    
    logging.info(f'Successfully converted DICOM series to NIfTI for {os.path.dirname(output_fname)}')
    

def convert_dicom_to_nifti(dicom_path: Union[str, List[str]], file_path: str, loader: str = 'sitk', **kwargs) -> None:
    """
    Converts a DICOM series to a NIfTI file.

    This function loads a DICOM series from a file or a directory, or a list of dicom files, using the 
    `load_dicom_images` function and saves it as a NIfTI file using the `save_itk_image_as_nifti_sitk` function.

    Args:
        dicom_path (Union[str, List[str]]): The path to the directory containing DICOM files, the path to a single
            DICOM file, or a list of dicom files.
        file_path (str): The path to save the output NIfTI file.
        **kwargs: Optional keyword arguments to pass to the `load_dicom_images` function.

    Returns:
        None

    Raises:
        RuntimeError: If the DICOM series could not be read or the NIfTI file could not be saved.
    """
    if loader=='sitk':
        image_itk = load_dicom_images(dicom_path, **kwargs)
        # print(f'affine in dicom image = {get_affine_from_itk_image(image_itk)}')
        save_itk_image_as_nifti_sitk(image_itk, file_path)
        # nii_image = nib.load(file_path)
        # print(f'nii_image.affine = {nii_image.affine}')
    elif loader=='pydicom':
        pydicom_data = load_dicom_images_pydicom(dicom_path, **kwargs)
        save_pydicom_to_nifti_nib(pydicom_data, file_path)

### Convert RTSS to nifti ###

def convert_dicom_rtss_to_nifti(dicom_image_files, rtss, output_fname, structure_map=None):
    """
    Convert DICOM RT Structure Set (RTSS) to NIfTI format.

    Args:
        dicom_image_files (list): List of DICOM image files.
        rtss (str): Path to the DICOM RTSS file.
        output_fname (str): Output filename for the NIfTI file.
        structure_map (dict, optional): Mapping of structure names to indices. Defaults to None.

    Returns:
        None
    """
    
    temp_dir = tempfile.mkdtemp()
    # copy all image files into the temp directory
    copy_file_safely(rtss, temp_dir)
    rtss_temp_file = os.path.join(temp_dir, os.path.basename(rtss))
    
    for image_file in dicom_image_files:
        copy_file_safely(image_file, temp_dir)
        
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=temp_dir,
        rt_struct_path=rtss_temp_file,
    )
    
    rtss_mask = fetch_rtstruct_roi_masks(rtstruct, structure_map)
    if isinstance(rtss_mask, type(None)):
        logging.info(f'found no masks for {os.path.basename(output_fname).split(".")[0]} when converting from dicom to nifti.')
    
    # To match with the dicom2nifti.dicom_series_to_nifti orientation
    rtss_mask = np.swapaxes(rtss_mask, 0, 1)
    
    rtss_mask = rtss_mask.astype(np.uint8)

    rtss_nii = nib.Nifti1Image(rtss_mask, affine=np.eye(4))
    
    nib.save(rtss_nii, output_fname)
    
    logging.info(f'Successfully converted DICOM RTSS to NIfTI for {os.path.dirname(rtss)}')


################# another way to convert rtss to nifti #################
###### This approach seems to give missing slices for some reason ######

# def convert_dicom_rtss_to_nifti(dicom_folder: str, dicom_rtss_path: str, output_nifti_path: str, 
#                   structure_map: Optional[Dict[int, List[str]]] = None) -> None:
#     """Convert DICOM RTSS file to a NIfTI file with structures mapped to specific values.

#     Args:
#         rtss_path (str): Path to the DICOM RT Structure Set file.
#         reference_image_dir (str): Path to the directory containing the reference DICOM image slices.
#         output_path (str): Path to save the output NIfTI file.
#         structure_map (Optional[Dict[int, List[str]]]): Mapping of values to structure names.

#     Raises:
#         Warning: If a structure is not found in the provided dictionary.
#     """
#     try:
#         rtss = pydicom.dcmread(dicom_rtss_path)
#     except Exception as e:
#         print(f"Error reading RTSS file: {e}")
#         return

#     try:
#         reader = sitk.ImageSeriesReader()
#         dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
#         reader.SetFileNames(dicom_names)
#         reference_image = reader.Execute()
#     except Exception as e:
#         print(f"Error reading reference image slices: {e}")
#         return

#     # Get the dimensions and spacing from the reference image
#     image_shape = reference_image.GetSize()
#     spacing = reference_image.GetSpacing()
#     origin = reference_image.GetOrigin()
#     direction = reference_image.GetDirection()
    
#     # Initialize an empty numpy array for the NIfTI file
#     nifti_array = np.zeros(image_shape[::-1], dtype=np.uint8)

#     # Create a reverse mapping for easier lookup
#     structure_to_value = {}
#     if structure_map:
#         for value, structures in structure_map.items():
#             for structure in structures:
#                 structure_to_value[structure] = value

#     # Process each structure in the RTSS file
#     for roi in rtss.ROIContourSequence:
#         roi_name = rtss.StructureSetROISequence[roi.ReferencedROINumber - 1].ROIName
#         if structure_map and roi_name not in structure_to_value:
#             print(f"Warning: Structure '{roi_name}' not found in the structure map.")
#             print(f"Consider including '{roi_name}' in the structure map.")
#             continue

#         value = structure_to_value.get(roi_name, 0)

#         for contour in roi.ContourSequence:
#             contour_data = contour.ContourData
#             contour_points = np.array(contour_data).reshape((-1, 3))
            
#            # Convert the contour points to voxel indices
#             voxel_indices = np.array([reference_image.TransformPhysicalPointToIndex(point) for point in contour_points])

#             # Create a mask for the contour and fill it
#             for z in np.unique(voxel_indices[:, 2]):
#                 slice_mask = np.zeros((image_shape[1], image_shape[0]), dtype=np.uint8)
#                 points_in_slice = voxel_indices[voxel_indices[:, 2] == z][:, :2]
#                 if len(points_in_slice) > 2:
#                     rr, cc = points_in_slice[:, 1], points_in_slice[:, 0]
#                     slice_mask[rr, cc] = 1
#                     filled_mask = binary_fill_holes(slice_mask)
#                     nifti_array[z, filled_mask] = value

#     # Fill 3D volumes
#     for i in range(nifti_array.shape[0]):
#         nifti_array[i] = binary_fill_holes(nifti_array[i])

#     # Create a SimpleITK image from the numpy array
#     try:
#         nifti_image = sitk.GetImageFromArray(nifti_array)
#         nifti_image.SetSpacing(spacing)
#         nifti_image.SetOrigin(origin)
#         nifti_image.SetDirection(direction)
#     except Exception as e:
#         print(f"Error creating NIfTI image: {e}")
#         return

#     # Write the NIfTI file
#     try:
#         sitk.WriteImage(nifti_image, output_nifti_path)
#         print(f"NIfTI file saved to {output_nifti_path}")
#     except Exception as e:
#         print(f"Error writing NIfTI file: {e}")