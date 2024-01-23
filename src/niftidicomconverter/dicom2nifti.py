import numpy as np
import nibabel as nib
import tempfile
import os
import logging

from typing import List, Union
from rt_utils import RTStructBuilder, RTStruct

from niftidicomconverter.dicomhandling import load_dicom_images, save_itk_image_as_nifti_sitk, get_affine_from_itk_image
from niftidicomconverter.dicomhandling_pydicom import load_dicom_images_pydicom, save_pydicom_to_nifti_nib
from niftidicomconverter.utils import copy_nifti_header

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
        None.

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


def fetch_mapped_rois(rtstruct: RTStruct, structure_map: dict) -> np.ndarray:

    masks = {}
    roi_names = rtstruct.get_roi_names()

    for roi_idx, roi_name in enumerate(roi_names):

        for structure_idx, structures in structure_map.items():

            mask_idx = int(structure_idx)

            if roi_name.lower() not in (_.lower() for _ in structures):
                continue

            logging.debug(f"\t-- Converting structure: {roi_name}")

            try:
                mask = rtstruct.get_roi_mask_by_name(roi_name)

                number_voxel = np.count_nonzero(mask)
                logging.debug(f"\tCounted number of voxel: {number_voxel}")

                if number_voxel == 0:
                    continue

                if mask_idx in masks:
                    mask = np.logical_or(mask, masks[mask_idx])

                masks[mask_idx] = mask

                break

            except Exception as e:
                logging.error(e)
                break

    if len(masks) == 0:
        return None

    shape = masks[list(masks.keys())[0]].shape
    shape = shape + (len(structure_map) + 1,)
    stacked_mask = np.zeros(shape, dtype=np.uint8)

    for idx, mask in masks.items():
        stacked_mask[:, :, :, idx] = mask.astype(np.uint8)

    # Set background
    background = np.sum(stacked_mask, axis=-1) == 0
    stacked_mask[..., 0] = background

    return stacked_mask

def fetch_rtstruct_roi_masks(
    rtstruct: RTStruct,
    structure_map: dict = None,
) -> np.ndarray:
    """
    Default structure list start from 1
    """

    if structure_map is not None:
        masks = fetch_mapped_rois(rtstruct, structure_map)

    else:
        masks = fetch_all_rois(rtstruct)

    return masks

def fetch_all_rois(rtstruct: RTStruct) -> np.ndarray:

    masks = []
    roi_names = rtstruct.get_roi_names()

    for roi_name in roi_names:
        mask = rtstruct.get_roi_mask_by_name(roi_name)
        masks.append(mask)

    if len(masks) == 0:
        return None

    flat_mask = np.sum(masks, axis=0) > 0

    return flat_mask

# def fetch_all_rois_variant_depr(rtstruct: RTStruct) -> np.ndarray:
#     """
#     Fetch all ROI masks from an RTStruct object and combine them into a 3D numpy array.

#     Args:
#         rtstruct (RTStruct): The RTStruct object containing the ROIs to be fetched.

#     Returns:
#         np.ndarray: A 3D numpy array containing the binary masks for all ROIs in the RTStruct object.
#             The array has shape (Z, Y, X, N), where Z, Y, and X are the dimensions of the image and N
#             is the number of ROIs, including the background.
#     """
#     masks = []
#     roi_names = rtstruct.get_roi_names()

#     for roi_name in roi_names:
#         mask = rtstruct.get_roi_mask_by_name(roi_name)
#         masks.append(mask)
#         print(roi_name)

#     if len(masks) == 0:
#         return None

#     background = np.sum(masks, axis=0) == 0

#     masks.insert(0, background)  # Add background to the mask

#     return np.stack(masks, axis=-1)

def convert_dicom_rtss_to_nifti(
    dicom_folder: str,
    dicom_rtss_path: str,
    output_nifti_path: str,
    structure_map: dict
) -> None:
    """
    Convert a DICOM RT Structure Set (RTSS) file to a NIfTI binary mask file.
    
    This function reads a DICOM RTSS file and converts the contours of structures in the image into a binary mask. The 
    binary mask is then saved as a NIfTI file that can be used for image segmentation or registration tasks.
    
    Args:
        dicom_folder (str): Path to the folder containing the DICOM images that the RTSS file corresponds to.
        dicom_rtss_path (str): Path to the DICOM RTSS file to convert.
        output_nifti_path (str): Path to save the output NIfTI file to.
    
    Returns:
        None.
    """

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=dicom_folder,
        rt_struct_path=dicom_rtss_path,
    )

    # rtss_mask = masks = fetch_all_rois(rtstruct)
    rtss_mask = fetch_rtstruct_roi_masks(rtstruct, structure_map)
    # import pdb; pdb.set_trace()
    rtss_mask = rtss_mask.astype(np.float32)

    # To match with the dicom2nifti.dicom_series_to_nifti orientation
    rtss_mask = np.swapaxes(rtss_mask, 0, 1)

    rtss_nii = nib.Nifti1Image(rtss_mask, affine=np.eye(4)) # note that the affine will be calculated from the header later, so don't need to calculate the affine here

    # this is to get the header of the original dicom
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmpfile = os.path.join(tmp_dir, 'image.nii')

        dicom_image_sitk = load_dicom_images(dicom_folder)
        save_itk_image_as_nifti_sitk(dicom_image_sitk, tmpfile)

        nifti_image_src = nib.load(tmpfile)

        nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)

    nib.save(nib_rtss, output_nifti_path)

    

