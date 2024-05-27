import numpy as np
import nibabel as nib
import tempfile
import os
import logging

from typing import List, Union, Dict, Optional
from rt_utils import RTStructBuilder, RTStruct

from niftidicomconverter.dicomhandling import load_dicom_images, save_itk_image_as_nifti_sitk, get_affine_from_itk_image
from niftidicomconverter.dicomhandling_pydicom import load_dicom_images_pydicom, save_pydicom_to_nifti_nib
from niftidicomconverter.utils import copy_nifti_header

import pydicom
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon

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


# def fetch_mapped_rois(rtstruct: RTStruct, structure_map: dict) -> np.ndarray:
#     """
#     Fetches and maps Regions of Interest (ROIs) from an RTStruct object to a specified structure map,
#     returning a stacked numpy array of masks corresponding to each structure.

#     This function iterates over ROI names in the RTStruct, checks if each ROI name matches any of the
#     structures specified in `structure_map`, and if so, extracts the mask for that ROI. Masks for
#     ROIs corresponding to the same structure index are combined using a logical OR operation. The
#     function returns a 4D numpy array, where the first three dimensions correspond to the spatial
#     dimensions of the masks, and the fourth dimension indexes the structures plus a background layer.
#     The background layer is set to 1 for voxels not belonging to any ROI.

#     Args:
#         rtstruct (RTStruct): An object representing the RT structure set, which must have methods
#             `get_roi_names()` and `get_roi_mask_by_name(roi_name)`.
#         structure_map (dict): A mapping from structure index (as integer keys) to lists of structure names
#             (as strings). The function maps each ROI to these indices based on matching names.

#     Returns:
#         np.ndarray: A 4D numpy array of uint8, where each slice along the fourth dimension represents a mask
#             for a different structure, including an additional background layer as the first index.

#     Raises:
#         Exception: Logs an error and breaks the current iteration if `get_roi_mask_by_name` raises an Exception.

#     Example:
#         ```python
#         rtstruct = load_rtstruct("path/to/dicom/folder")
#         structure_map = {0: ["Structure1", "Structure2"], 1: ["Structure3"]}
#         masks = fetch_mapped_rois(rtstruct, structure_map)
#         ```
#     """

#     masks = {}
#     roi_names = rtstruct.get_roi_names()

#     for roi_idx, roi_name in enumerate(roi_names):

#         for structure_idx, structures in structure_map.items():

#             mask_idx = int(structure_idx)

#             if roi_name.lower() not in (_.lower() for _ in structures):
#                 continue

#             logging.debug(f"\t-- Converting structure: {roi_name}")

#             try:
#                 mask = rtstruct.get_roi_mask_by_name(roi_name)

#                 number_voxel = np.count_nonzero(mask)
#                 logging.debug(f"\tCounted number of voxel: {number_voxel}")

#                 if number_voxel == 0:
#                     continue

#                 if mask_idx in masks:
#                     mask = np.logical_or(mask, masks[mask_idx])

#                 masks[mask_idx] = mask

#                 break

#             except Exception as e:
#                 logging.error(e)
#                 break

#     if len(masks) == 0:
#         return None

#     shape = masks[list(masks.keys())[0]].shape
#     shape = shape + (len(structure_map) + 1,)
#     stacked_mask = np.zeros(shape, dtype=np.uint8)

#     for idx, mask in masks.items():
#         stacked_mask[:, :, :, idx] = mask.astype(np.uint8)

#     # Set background
#     background = np.sum(stacked_mask, axis=-1) == 0
#     stacked_mask[..., 0] = background

#     return stacked_mask

# def fetch_rtstruct_roi_masks(
#     rtstruct: RTStruct,
#     structure_map: dict = None,
# ) -> np.ndarray:
#     """
#     Fetches ROI masks from an RTStruct object, optionally mapping them according to a given structure map.
#     If a structure map is provided, it uses `fetch_mapped_rois` to fetch and map ROIs to the specified
#     structure map. If no structure map is provided, it fetches all ROIs using `fetch_all_rois`.

#     Args:
#         rtstruct (RTStruct): An object representing the RT structure set, which must have methods
#             `get_roi_names()` and `get_roi_mask_by_name(roi_name)`.
#         structure_map (dict, optional): A mapping from structure index (as integer keys) to lists of
#             structure names (as strings). If provided, ROIs are mapped to these indices based on matching names.
#             If None, all ROI masks are fetched without mapping.

#     Returns:
#         np.ndarray: A numpy array of masks. The array is 4D if `structure_map` is provided and mapping
#             is performed, otherwise, a 3D array representing a flat mask of all ROIs.

#     Example:
#         ```python
#         rtstruct = load_rtstruct("path/to/dicom/folder")
#         structure_map = {0: ["Structure1", "Structure2"], 1: ["Structure3"]}
#         masks = fetch_rtstruct_roi_masks(rtstruct, structure_map)
#         ```
#     """
#     if structure_map is not None:
#         masks = fetch_mapped_rois(rtstruct, structure_map)

#     else:
#         masks = fetch_all_rois(rtstruct)

#     return masks

# def fetch_all_rois(rtstruct: RTStruct) -> np.ndarray:
#     """
#     Fetches all ROI masks from an RTStruct object and combines them into a single flat mask.
#     This function iterates over all ROI names in the RTStruct, retrieves each ROI's mask,
#     and aggregates these masks into a single flat mask where each voxel's value indicates
#     the presence (1) or absence (0) of any ROI.

#     Args:
#         rtstruct (RTStruct): An object representing the RT structure set, which must have methods
#             `get_roi_names()` and `get_roi_mask_by_name(roi_name)`.

#     Returns:
#         np.ndarray: A 3D numpy array of uint8, representing the aggregated presence of all ROIs.
#             If no ROIs are present, returns None.

#     Example:
#         ```python
#         rtstruct = load_rtstruct("path/to/dicom/folder")
#         flat_mask = fetch_all_rois(rtstruct)
#         ```
#     """


#     masks = []
#     roi_names = rtstruct.get_roi_names()

#     for roi_name in roi_names:
#         mask = rtstruct.get_roi_mask_by_name(roi_name)
#         masks.append(mask)

#     if len(masks) == 0:
#         return None

#     flat_mask = np.sum(masks, axis=0) > 0

#     return flat_mask

# # def fetch_all_rois_variant_depr(rtstruct: RTStruct) -> np.ndarray:
# #     """
# #     Fetch all ROI masks from an RTStruct object and combine them into a 3D numpy array.

# #     Args:
# #         rtstruct (RTStruct): The RTStruct object containing the ROIs to be fetched.

# #     Returns:
# #         np.ndarray: A 3D numpy array containing the binary masks for all ROIs in the RTStruct object.
# #             The array has shape (Z, Y, X, N), where Z, Y, and X are the dimensions of the image and N
# #             is the number of ROIs, including the background.
# #     """
# #     masks = []
# #     roi_names = rtstruct.get_roi_names()

# #     for roi_name in roi_names:
# #         mask = rtstruct.get_roi_mask_by_name(roi_name)
# #         masks.append(mask)
# #         print(roi_name)

# #     if len(masks) == 0:
# #         return None

# #     background = np.sum(masks, axis=0) == 0

# #     masks.insert(0, background)  # Add background to the mask

# #     return np.stack(masks, axis=-1)


# def convert_dicom_rtss_to_nifti(
#     dicom_folder: str,
#     dicom_rtss_path: str,
#     output_nifti_path: str,
#     structure_map: dict
# ) -> None:
#     """
#     Converts a DICOM RT Structure Set (RTSS) file into a NIfTI format binary mask file,
#     applying a structure map to filter and organize the ROI masks into the output file.

#     This function processes a specified DICOM RTSS file along with its corresponding DICOM series,
#     creating a binary mask for each ROI defined in the RTSS and mapping these to a specified
#     structure map. The masks are then saved as a NIfTI file.

#     Args:
#         dicom_folder (str): Path to the folder containing the DICOM series associated with the RTSS.
#         dicom_rtss_path (str): Path to the DICOM RT Structure Set file to be converted.
#         output_nifti_path (str): Destination path where the resulting NIfTI file will be saved.
#         structure_map (dict): A mapping from structure indices (integer keys) to lists of structure
#             names (strings) that specifies how ROI masks should be organized in the NIfTI file.

#     Returns:
#         None: The function saves the generated NIfTI file to the specified output path but does not return any value.
#     """


#     rtstruct = RTStructBuilder.create_from(
#         dicom_series_path=dicom_folder,
#         rt_struct_path=dicom_rtss_path,
#     )

#     # rtss_mask = masks = fetch_all_rois(rtstruct)
#     rtss_mask = fetch_rtstruct_roi_masks(rtstruct, structure_map)
#     # import pdb; pdb.set_trace()
#     rtss_mask = rtss_mask.astype(np.float32)

#     # To match with the dicom2nifti.dicom_series_to_nifti orientation
#     rtss_mask = np.swapaxes(rtss_mask, 0, 1)

#     rtss_nii = nib.Nifti1Image(rtss_mask, affine=np.eye(4)) # note that the affine will be calculated from the header later, so don't need to calculate the affine here

#     # this is to get the header of the original dicom
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tmpfile = os.path.join(tmp_dir, 'image.nii')

#         dicom_image_sitk = load_dicom_images(dicom_folder)
#         save_itk_image_as_nifti_sitk(dicom_image_sitk, tmpfile)

#         nifti_image_src = nib.load(tmpfile)

#         nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)

#     nib.save(nib_rtss, output_nifti_path)

################# another way to convert rtss to nifti #################

import pydicom
import SimpleITK as sitk
import numpy as np
from typing import Dict, List, Optional
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon

# def convert_dicom_rtss_to_nifti(
#     dicom_folder: str,
#     dicom_rtss_path: str,
#     output_nifti_path: str,
#     structure_map: dict

def convert_dicom_rtss_to_nifti(dicom_folder: str, dicom_rtss_path: str, output_nifti_path: str, 
                  structure_map: Optional[Dict[int, List[str]]] = None) -> None:
    """Convert DICOM RTSS file to a NIfTI file with structures mapped to specific values.

    Args:
        rtss_path (str): Path to the DICOM RT Structure Set file.
        reference_image_dir (str): Path to the directory containing the reference DICOM image slices.
        output_path (str): Path to save the output NIfTI file.
        structure_map (Optional[Dict[int, List[str]]]): Mapping of values to structure names.

    Raises:
        Warning: If a structure is not found in the provided dictionary.
    """
    try:
        rtss = pydicom.dcmread(dicom_rtss_path)
    except Exception as e:
        print(f"Error reading RTSS file: {e}")
        return

    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        reader.SetFileNames(dicom_names)
        reference_image = reader.Execute()
    except Exception as e:
        print(f"Error reading reference image slices: {e}")
        return

    # Get the dimensions and spacing from the reference image
    image_shape = reference_image.GetSize()
    spacing = reference_image.GetSpacing()
    origin = reference_image.GetOrigin()
    direction = reference_image.GetDirection()
    
    # Initialize an empty numpy array for the NIfTI file
    nifti_array = np.zeros(image_shape[::-1], dtype=np.uint8)

    # Create a reverse mapping for easier lookup
    structure_to_value = {}
    if structure_map:
        for value, structures in structure_map.items():
            for structure in structures:
                structure_to_value[structure] = value

    # Process each structure in the RTSS file
    for roi in rtss.ROIContourSequence:
        roi_name = rtss.StructureSetROISequence[roi.ReferencedROINumber - 1].ROIName
        if structure_map and roi_name not in structure_to_value:
            print(f"Warning: Structure '{roi_name}' not found in the structure map.")
            print(f"Consider including '{roi_name}' in the structure map.")
            continue

        value = structure_to_value.get(roi_name, 0)

        for contour in roi.ContourSequence:
            contour_data = contour.ContourData
            contour_points = np.array(contour_data).reshape((-1, 3))
            
           # Convert the contour points to voxel indices
            voxel_indices = np.array([reference_image.TransformPhysicalPointToIndex(point) for point in contour_points])

            # Create a mask for the contour and fill it
            for z in np.unique(voxel_indices[:, 2]):
                slice_mask = np.zeros((image_shape[1], image_shape[0]), dtype=np.uint8)
                points_in_slice = voxel_indices[voxel_indices[:, 2] == z][:, :2]
                if len(points_in_slice) > 2:
                    rr, cc = points_in_slice[:, 1], points_in_slice[:, 0]
                    slice_mask[rr, cc] = 1
                    filled_mask = binary_fill_holes(slice_mask)
                    nifti_array[z, filled_mask] = value

    # Fill 3D volumes
    for i in range(nifti_array.shape[0]):
        nifti_array[i] = binary_fill_holes(nifti_array[i])

    # Create a SimpleITK image from the numpy array
    try:
        nifti_image = sitk.GetImageFromArray(nifti_array)
        nifti_image.SetSpacing(spacing)
        nifti_image.SetOrigin(origin)
        nifti_image.SetDirection(direction)
    except Exception as e:
        print(f"Error creating NIfTI image: {e}")
        return

    # Write the NIfTI file
    try:
        sitk.WriteImage(nifti_image, output_nifti_path)
        print(f"NIfTI file saved to {output_nifti_path}")
    except Exception as e:
        print(f"Error writing NIfTI file: {e}")