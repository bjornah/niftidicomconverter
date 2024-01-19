import SimpleITK as sitk
import numpy as np
from rt_utils import RTStructBuilder
import tempfile
import glob
from typing import Optional, Union, List

from .utils import copy_file_safely, axes_swapping, get_binary_rtss, find_clusters_itk

def nifti_rtss_to_dicom_rtss(
    nifti_rtss,
    original_dicom_folder, 
    output_dicom_rtss, 
    inference_threshold: float=0.5, 
    new_spacing: Union[tuple, str]='original',
    dicom_list: Optional[List]=None,
    clustering_threshold: Optional[int]=None
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
        # logging.info(f'number of disjoint clusters/tumours in prediction: {n_target}')
        
    nda = sitk.GetArrayFromImage(label_image)
    nda = axes_swapping(nda)
    # nda = nda[::-1,:,:]
    
    # logging.debug(f'unique labels in image: {np.unique(nda)}')

    if dicom_list is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for dicom in dicom_list:
                copy_file_safely(
                    tmp_dir=tmp_dir,
                    src=dicom,
                    dst_naming=dicom.split('/')[-1],
                )

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
        rtstruct.add_roi(
            mask=mask, 
            color=[100, 150, int(255/n_target*idx)], 
            name=f"MET nr {idx}"
            )
    # logging.debug(f'There are currently {small_clusters} clusters with fewer than 10 pixels.')

    rtstruct.save(output_dicom_rtss)

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
