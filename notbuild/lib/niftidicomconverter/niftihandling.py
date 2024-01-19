import SimpleITK as sitk
import nibabel as nib
from typing import Union

def read_nifti_file_sitk(file_path: str) -> sitk.Image:
    """
    Reads a NIfTI file using SimpleITK and returns it as a SimpleITK image object.

    Args:
        file_path (str): The path to the NIfTI file to be read.

    Returns:
        A SimpleITK image object representing the NIfTI file.

    Raises:
        RuntimeError: If the NIfTI file could not be read or is invalid.
    """
    try:
        image = sitk.ReadImage(file_path)
    except RuntimeError as e:
        raise RuntimeError(f"Could not read NIfTI file '{file_path}': {str(e)}")
    
    if not sitk.Image(image).GetNumberOfComponentsPerPixel() == 1:
        raise RuntimeError(f"Invalid NIfTI file '{file_path}': contains more than one component per pixel")
    
    return image

def read_nifti_file_nib(file_path: str) -> nib.nifti1.Nifti1Image:
    """
    Loads a NIfTI file using nibabel and returns it as a nibabel Nifti1Image object.

    Args:
        file_path (str): The path to the NIfTI file to be loaded.

    Returns:
        A nibabel Nifti1Image object representing the NIfTI file.

    Raises:
        FileNotFoundError: If the NIfTI file is not found.
        nib.filebasedimages.ImageFileError: If the NIfTI file is not a valid NIfTI file.
    """
    try:
        image = nib.load(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find NIfTI file '{file_path}'")
    except nib.filebasedimages.ImageFileError as e:
        raise nib.filebasedimages.ImageFileError(f"Invalid NIfTI file '{file_path}': {str(e)}")
    
    return image
