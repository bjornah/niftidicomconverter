import SimpleITK as sitk
import numpy as np
from rt_utils import RTStructBuilder
import tempfile
import glob
import logging
from typing import Optional, Union, List
import pydicom
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure, zoom

from .niftihandling import detection_and_segmentation_binarization_2D
from .utils import copy_file_safely, axes_swapping, get_binary_rtss, find_clusters_itk

def nifti_rtss_to_dicom_rtss(
    nifti_rtss,
    original_dicom_folder, 
    output_dicom_rtss, 
    inference_threshold: float=0.5, 
    new_spacing: Union[tuple, str]='original',
    dicom_list: Optional[List]=None,
    clustering_threshold: Optional[int]=None,
    roi_colors: Optional[List]=None,
    color_base: Optional[List]=[100, 150, 0]
    ):
    """
    Convert a NIfTI binary mask to a DICOM RT Structure Set (RTSS) file.
    
    This function takes a binary mask in NIfTI format and converts it to a DICOM RTSS file that can be used for radiation
    therapy planning. The RTSS file will contain regions of interest (ROIs) corresponding to the different labels in the
    binary mask. The function also has options for adjusting the spacing of the mask, clustering small ROIs, and working
    with a list of DICOM files.
    
    Args:
        nifti_rtss (str or np.ndarray): Path to the NIfTI binary mask file to convert, or a numpy array containing the
            binary mask data.
        original_dicom_folder (str): Path to the folder containing the original DICOM images that the NIfTI mask
            corresponds to.
        output_dicom_rtss (str): Path to save the output DICOM RTSS file to.
        inference_threshold (float, optional): Threshold value for the binary mask (default 0.5).
        new_spacing (tuple or str, optional): Desired spacing for the mask voxels, specified as a tuple of (x, y, z)
            values. If set to 'original' (default), the spacing will be set to match the original DICOM images.
        dicom_list (list of str, optional): List of paths to individual DICOM files instead of the entire DICOM series
            in `original_dicom_folder`.
        clustering_threshold (int, optional): Maximum number of ROIs to keep in the output RTSS file (default None).
    
    Returns:
        None.
    """
    
    
    if new_spacing=='original':
        if dicom_list is None:
            dicom_image = sitk.ReadImage(glob.glob(original_dicom_folder+'/*dcm')[0])
        else:
            dicom_image = sitk.ReadImage(dicom_list[0])
        new_spacing = dicom_image.GetSpacing()
    
    # logging.debug(f'nifti_rtss = {nifti_rtss}')
    rtss_binary = get_binary_rtss(nifti_rtss, inference_threshold, new_spacing)

    if clustering_threshold:
        label_image, n_target = find_clusters_itk(rtss_binary, max_object=clustering_threshold)
    else:
        label_image, n_target = find_clusters_itk(rtss_binary)
        # logging.info(f'number of disjoint clusters/tumours in prediction: {n_target}')
        
    nda = sitk.GetArrayFromImage(label_image)
    nda = axes_swapping(nda)
    # nda = nda[::-1,:,:]
    
    # logging.debug(f'unique labels in image: {np.unique(nda)}')

    if dicom_list is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for dicom in dicom_list:
                copy_file_safely(
                    file_path=dicom,
                    target_dir=tmp_dir
                )
                
                # copy_file_safely(
                #     tmp_dir=tmp_dir,
                #     src=dicom,
                #     dst_naming=dicom.split('/')[-1],
                # )

            rtstruct = RTStructBuilder.create_new(
                            dicom_series_path=tmp_dir,
                            )
    else:
        rtstruct = RTStructBuilder.create_new(
                        dicom_series_path=original_dicom_folder,
                        )


    small_clusters = 0
    for idx in range(1,n_target+1): # 0 is background, so don't use that label as ROI
        ROI = np.where(nda==idx, 1, 0)
        mask = np.ma.make_mask(ROI)
        if mask.sum()<10:
            small_clusters+1
        if clustering_threshold:
            if mask.sum()<clustering_threshold:
                logging.info(f'Cluster {idx} has fewer than {clustering_threshold} voxels, excluding it.')
                continue

        if roi_colors is None:
            colour = [color_base[0], color_base[1], int(255/n_target*idx)]
        elif isinstance(roi_colors[0], list):
            colour = roi_colors[idx-1]
        else:
            colour = roi_colors

        rtstruct.add_roi(
            mask=mask, 
            color=colour, 
            name=f"MET nr {idx}"
            )
    # logging.debug(f'There are currently {small_clusters} clusters with fewer than 10 pixels.')

    rtstruct.save(output_dicom_rtss)


################################################################################
### Another way of doing it ###
################################################################################

def rescale_image_to_dicom(image: np.ndarray, dicom_file: str, orig_spacing = np.array([1, 1, 1])) -> np.ndarray:
    """
    Rescales the input image to match the pixel spacing and slice thickness of a DICOM file.

    Args:
        image (np.ndarray): The input image to be rescaled.
        dicom_file (str): The path to the DICOM file used for rescaling.
        orig_spacing (np.ndarray, optional): The original pixel spacing and slice thickness of the image. 
            Defaults to np.array([1, 1, 1]).

    Returns:
        np.ndarray: The rescaled image.
    """
    # Load the DICOM file
    dicom = pydicom.dcmread(dicom_file)

    # Get the pixel spacing and slice thickness from the DICOM file
    dicom_spacing = np.array(list(dicom.PixelSpacing) + [dicom.SliceThickness], dtype=np.float32) # want order x,y,z since this is the order of axes in the nifti file. I think.

    # Calculate the zoom factor
    zoom_factor = dicom_spacing / orig_spacing

    # If the zoom factor is 1 in all dimensions, return the original image
    if np.allclose(zoom_factor, 1):
        logging.info(f'No rescaling needed, zoom factor: {zoom_factor}. Returning original image.')
        return image
    
    # Rescale the image
    rescaled_image = zoom(image, zoom_factor)

    return rescaled_image


############## The following function does not work and I do not know why ##############
############## All contours end up in the wrong place with this function  ##############
############## and it is not the transposition that is the problem.       ##############

# def prob_map_to_dicom_rtss(pred, dicom_folder, output_file, detection_threshold=0.5, binarization_threshold=0.5, roi_colors=None, color_base=[100, 150, 0], clustering_threshold=None):
#     """
#     Converts a probability map to a DICOM RT Structure Set (RTSS).

#     This function takes a probability map (either as a numpy array or a NIfTI file), 
#     rescales it to match the dimensions of a DICOM series, binarizes it using the provided 
#     thresholds, clusters the binarized map into separate regions of interest (ROIs), and 
#     adds each ROI to a new RTSS. The RTSS is then saved to a file.

#     Parameters:
#     pred (np.ndarray or str): The probability map to convert. Can be a numpy array or 
#                               the path to a NIfTI file.
#     dicom_folder (str): The path to the folder containing the DICOM series that the 
#                         probability map should be rescaled to match.
#     output_file (str): The path where the RTSS should be saved.
#     detection_threshold (float, optional): The threshold for detection in the binarization 
#                                            process. Defaults to 0.5.
#     binarization_threshold (float, optional): The threshold for binarization in the 
#                                               binarization process. Defaults to 0.5.
#     roi_colors (list or tuple, optional): The colors to use for the ROIs. Specified as
#                                             a list of [R, G, B] values for each ROI.
#                                             If None, the default color base is used.
#                                             Defaults to None.
#     color_base (list or tuple, optional): The base color to use for the ROIs,
#                                             specified as [R, G, B], if roi_colors is None.
#                                             Defaults to [100, 150, 0].
#     clustering_threshold (int, optional): The minimum number of voxels required to keep ROIs
#                                             in the RTSS. If an ROI has fewer voxels than this
#                                             threshold, it will be removed. Defaults to None.

#     Returns:
#     None
#     """
#     if not isinstance(pred, np.ndarray):
#         pred = nib.load(pred).get_fdata()

#     dicom_list = glob.glob(dicom_folder+'/*dcm')

#     # Create new RT Struct. Requires the DICOM series path for the RT Struct.
#     rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_folder)

#     # rescale pred to the same dimensions as the dicom images
#     pred = rescale_image_to_dicom(pred, dicom_list[0])

#     # binarise the pred
#     pred = detection_and_segmentation_binarization_2D(prob_map = pred, detection_threshold = detection_threshold, binarization_threshold = binarization_threshold)

#     # cluster the binarised pred
#     labeled, n_target = label(pred, structure=generate_binary_structure(3, 3))

#     labeled = np.swapaxes(labeled, 0, 1) # need this bullshit


#     small_clusters = 0
#     for idx in range(1,n_target+1): # 0 is background, so don't use that label as ROI
#         ROI = np.where(labeled==idx, 1, 0)
#         mask = np.ma.make_mask(ROI)
#         if mask.sum()<10:
#             small_clusters+1
#         if clustering_threshold:
#             if mask.sum()<clustering_threshold:
#                 print(f'Cluster {idx} has fewer than {clustering_threshold} voxels, excluding it.')
#                 continue

#         if roi_colors is None:
#             colour = [color_base[0], color_base[1], int(255/n_target*idx)]
#         elif isinstance(roi_colors[0], list):
#             colour = roi_colors[idx-1]
#         else:
#             colour = roi_colors

#         rtstruct.add_roi(
#             mask=mask, 
#             color=colour, 
#             name=f"MET nr {idx}"
#             )
#     print(f'found {idx} structures')
        
#     rtstruct.save(output_file)

# def copy_nifti_header(src: nib.Nifti1Image, dst: nib.Nifti1Image) -> nib.Nifti1Image:
#     """Copy header from src to dst while perserving the dst data."""
#     data = dst.get_fdata()
#     return nib.nifti1.Nifti1Image(data, None, header=src.header)

# def load_nifti_rtss(nifti_pred):
#     rtss = sitk.ReadImage(nifti_pred)
    
#     binary_filter = sitk.BinaryThresholdImageFilter()
#     binary_filter.SetLowerThreshold(0.5)
#     binary_filter.SetUpperThreshold(1)
#     rtss_binary = binary_filter.Execute(rtss)

#     nifti_rtss = sitk.GetArrayFromImage(rtss_binary)

#     return nifti_rtss
