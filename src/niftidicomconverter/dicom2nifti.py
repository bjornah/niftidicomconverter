import numpy as np
import nibabel as nib
from typing import List, Union

from rt_utils import RTStructBuilder

from niftidicomconverter.dicomhandling import load_dicom_images, save_itk_image_as_nifti_sitk, get_affine_from_itk_image
from niftidicomconverter.dicomhandling_pydicom import load_dicom_images_pydicom, save_pydicom_to_nifti_nib
from niftidicomconverter.utils import fetch_all_rois

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
        save_itk_image_as_nifti_sitk(image_itk, file_path)
    elif loader=='pydicom':
        pydicom_data = load_dicom_images_pydicom(dicom_path, **kwargs)
        save_pydicom_to_nifti_nib(pydicom_data, file_path)


# this does not work. Does not translate all the relevant structures!
def convert_dicom_rtss_to_nifti(
    dicom_folder: str,
    dicom_rtss_path: str,
    output_nifti_path: str,
) -> None:
    """
    # this does not work. Does not translate all the relevant structures!

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

    rtss_mask = masks = fetch_all_rois(rtstruct)
    rtss_mask = rtss_mask.astype(np.float32)

    # To match with the dicom2nifti.dicom_series_to_nifti orientation
    rtss_mask = np.swapaxes(rtss_mask, 0, 1)


    dicom_image = load_dicom_images(dicom_folder)
    affine = get_affine_from_itk_image(dicom_image)

    # rtss_nii = nib.Nifti1Image(rtss_mask, affine=np.eye(4))
    rtss_nii = nib.Nifti1Image(rtss_mask, affine=affine)

    nib.save(rtss_nii, output_nifti_path)
    
