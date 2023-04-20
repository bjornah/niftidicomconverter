import os
import pydicom
import nibabel as nib
import numpy as np
from rt_utils import RTStructBuilder
from .utils import fetch_all_rois

import os
import pydicom
import numpy as np
import nibabel as nib


def dicom_folder_to_nifti(dicom_folder: str, output_nifti: str) -> None:
    """
    Convert a stack of DICOM images in a folder to a NIfTI file. Includes metadata from the DICOM files. 
    Ignores RTSS files in the same folder.

    Args:
        dicom_folder (str): Path to the folder containing the DICOM images.
        output_nifti (str): Path to save the output NIfTI file to.

    Returns:
        None.
    """

    # Load the DICOM images and metadata
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if (f.endswith('.dcm') and pydicom.read_file(f, force=True).Modality != 'RTSTRUCT')]
    dicom_data = [pydicom.read_file(f, force=True) for f in dicom_files]

    # Check that all DICOM images have the same image position and spacing
    assert all(d.ImagePositionPatient == dicom_data[0].ImagePositionPatient for d in dicom_data), \
        'DICOM images have different ImagePositionPatient values'
    assert all(d.PixelSpacing == dicom_data[0].PixelSpacing for d in dicom_data), \
        'DICOM images have different PixelSpacing values'

    # Create the NIfTI image from the DICOM data
    nifti_image = nib.Nifti1Image(np.stack([d.pixel_array for d in dicom_data]), dicom_data[0].ImagePositionPatient)
    nifti_header = nifti_image.header.copy()

    # Add DICOM header information to the NIfTI header
    for key in dicom_data[0].dir():
        if key not in nifti_header.keys():
            value = getattr(dicom_data[0], key, '')
            if isinstance(value, pydicom.uid.UID):
                value = str(value)
            nifti_header[key] = value

    # Save the NIfTI image to file
    nib.save(nifti_image, output_nifti, header=nifti_header, check=False)


def convert_single_dicoms_to_nifti(input_folder: str, output_folder: str) -> None:
    """
    Convert DICOM files in the input folder to NIfTI files in the output folder, and copy metadata from DICOM files to
    Nifti file headers. Note that this assumes that all files in the input folder are from different series and as such 
    will be converted into individual NIfTI files.

    Args:
        input_folder (str): Path to the input DICOM folder.
        output_folder (str): Path to the output NIfTI folder.

    Returns:
        None.
    """
    # Input and output folders

    # Get list of DICOM files
    dicom_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.dcm')]

    for dicom_file in dicom_files:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_file)
        
        # Create NIfTI image from DICOM data
        nifti_image = nib.nifti1.Nifti1Image(dicom_data.pixel_array, dicom_data.affine)
        
        # Add DICOM header information to NIfTI header
        nifti_header = nifti_image.header.copy()
        for key in dicom_data.dir():
            if key not in nifti_header.keys():
                value = getattr(dicom_data, key, '')
                if isinstance(value, pydicom.uid.UID):
                    value = str(value)
                nifti_header[key] = value
        
        # Save NIfTI image to file
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(dicom_file))[0] + '.nii.gz')
        nib.save(nifti_image, output_file)


def process_dicom_rtss_to_nifti(
    dicom_folder: str,
    dicom_rtss_path: str,
    output_nifti_path: str,
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

    rtss_mask = masks = fetch_all_rois(rtstruct)
    rtss_mask = rtss_mask.astype(np.float32)

    # To match with the dicom2nifti.dicom_series_to_nifti orientation
    rtss_mask = np.swapaxes(rtss_mask, 0, 1)

    rtss_nii = nib.Nifti1Image(rtss_mask, affine=np.eye(4))
    nib.save(rtss_nii, output_nifti_path)
