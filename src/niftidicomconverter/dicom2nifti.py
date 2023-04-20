import os
import pydicom
import nibabel as nib
import numpy as np
from rt_utils import RTStructBuilder
from .utils import fetch_all_rois

import pydicom
import numpy as np
import SimpleITK as sitk    
    
    

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
        
        
        # Try to get the affine matrix from the DICOM data, fallback to computing it from scratch
        if hasattr(dicom_data, 'affine'):
            affine = dicom_data.affine
        elif (
            hasattr(dicom_data, 'ImageOrientationPatient') and 
            hasattr(dicom_data, 'ImagePositionPatient') and 
            hasattr(dicom_data, 'PixelSpacing')
        ):
            orientation = np.array(dicom_data.ImageOrientationPatient).reshape((2, 3))
            position = np.array(dicom_data.ImagePositionPatient)
            spacing = np.array(dicom_data.PixelSpacing)
            direction = np.cross(orientation, orientation[1])
            affine = np.eye(4)
            affine[:3, :3] = orientation * spacing
            affine[:3, 3] = position
            affine[3, :3] = direction
            affine = np.linalg.inv(affine)
        else:
            affine = np.eye(4)

        # Create the NIfTI image from the DICOM data
        nifti_image = nib.nifti1.Nifti1Image(dicom_data.pixel_array, affine)
        
        # nifti_image = nib.Nifti1Image(np.stack([d.pixel_array for d in dicom_data]), affine)
        
        # Create NIfTI image from DICOM data
        # nifti_image = nib.nifti1.Nifti1Image(dicom_data.pixel_array, dicom_data.affine)
        
        # Add DICOM header information to NIfTI header
        nifti_header = nifti_image.header
        
        for key in dicom_data.dir():
            if key in nifti_header.keys():
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




#############################################



# def dicom_folder_to_nifti_2(dicom_folder: str, output_nifti: str) -> None:
#     """
#     Convert a stack of DICOM images in a folder to a NIfTI file. Includes metadata from the DICOM files. 
#     Ignores RTSS files in the same folder.

#     Args:
#         dicom_folder (str): Path to the folder containing the DICOM images.
#         output_nifti (str): Path to save the output NIfTI file to.

#     Returns:
#         None.
#     """

#     # Load the DICOM images and metadata    
#     dicom_files = [f_i for f_i in [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')] if pydicom.read_file(f_i, force=True).Modality != 'RTSTRUCT']
#     dicom_data = [pydicom.read_file(f, force=True) for f in dicom_files]

#     assert all(d.PixelSpacing == dicom_data[0].PixelSpacing for d in dicom_data), \
#         'DICOM images have different PixelSpacing values'

#     # Sort the slices along the z-axis
#     dicom_data.sort(key=lambda x: x.ImagePositionPatient[2])

#     # Extract the pixel arrays
#     pixel_arrays = [d.pixel_array for d in dicom_data]

#     # Resample the image so that all slices have the same increments if necessary
#     z_positions = [d.ImagePositionPatient[2] for d in dicom_data]
#     z_spacing = np.diff(z_positions)
#     if not np.allclose(z_spacing, z_spacing[0]):
#         from scipy.ndimage import zoom

#         z_scale = len(z_spacing) / (len(z_spacing) + 1)
#         pixel_arrays_resampled = []
#         for idx, pixel_array in enumerate(pixel_arrays):
#             pixel_arrays_resampled.append(zoom(pixel_array, z_scale))
#         pixel_arrays = pixel_arrays_resampled

#     # Create a 3D array of the pixel arrays
#     pixel_array_3d = np.stack(pixel_arrays)

#     # Calculate the affine transformation
#     # sample_dicom = dicom_data[0]
#     # orientation = np.array(sample_dicom.ImageOrientationPatient).reshape((2, 3))
#     # position = np.array(sample_dicom.ImagePositionPatient)
#     # spacing = np.array(sample_dicom.PixelSpacing)
#     # z_positions = [d.ImagePositionPatient[2] for d in dicom_data]
#     # z_spacing = np.diff(z_positions)
#     # slice_thickness = np.mean(z_spacing)
#     # direction = np.cross(orientation[0, :], orientation[1, :])
#     # affine = np.eye(4)
#     # affine[:3, :3] = np.column_stack([orientation[0, :] * spacing[0], orientation[1, :] * spacing[1], direction * slice_thickness])
#     # affine[:3, 3] = position
    
#     affine = calculate_affine(dicom_files)

#     # Create the NIfTI image from the DICOM data
#     nifti_image = nib.Nifti1Image(pixel_array_3d, affine)

#     # fix header
#     nifti_header = nifti_image.header

#     # Add DICOM header information to the NIfTI header
#     for key in dicom_data[0].dir():
#         if key in nifti_header.keys():
#             value = getattr(dicom_data[0], key, '')
#             if isinstance(value, pydicom.uid.UID):
#                 value = str(value)
#             nifti_header[key] = value

#     # Save the NIfTI image to file
#     nib.save(nifti_image, output_nifti)


# def convert_dicom_to_nifti(dicom_folder, output_nifti_file):
#     """
#     Convert all non-RTSTRUCT DICOM images in a folder to a single NIfTI file.

#     Parameters
#     ----------
#     dicom_folder : str
#         Path to the folder containing the DICOM files.
#     output_nifti_file : str
#         Path to the output NIfTI file.

#     Notes
#     -----
#     This function assumes all non-RTSTRUCT DICOM files in the folder belong to
#     the same series and have the same dimensions. Also, the function uses the
#     InstanceNumber attribute for sorting slices. This might not work correctly
#     for all cases. You may need to adjust the sorting method based on your
#     specific data.
#     """
    
#     # Get all DICOM files in the folder
#     dicom_files = glob(os.path.join(dicom_folder, '*.dcm'))

#     # Read the first non-RTSTRUCT DICOM file to get some metadata
#     sample_dicom = None
#     for dicom_file in dicom_files:
#         dicom_data = pydicom.dcmread(dicom_file)
#         if dicom_data.Modality != 'RTSTRUCT':
#             sample_dicom = dicom_data
#             break

#     if not sample_dicom:
#         print("No non-RTSTRUCT DICOM files found in the folder.")
#         return

# #     # Check that all DICOM images have the same image position and spacing
# #     assert all(d.ImagePositionPatient == dicom_data[0].ImagePositionPatient for d in dicom_data), \
# #         'DICOM images have different ImagePositionPatient values'
# #     assert all(d.PixelSpacing == dicom_data[0].PixelSpacing for d in dicom_data), \
# #         'DICOM images have different PixelSpacing values'

#     # Create an empty 3D numpy array to store the image data
#     img_shape = (int(sample_dicom.Rows), int(sample_dicom.Columns), len(dicom_files))
#     img_data = np.zeros(img_shape, dtype=sample_dicom.pixel_array.dtype)

#     # Iterate through the DICOM files and add the pixel data to the 3D numpy array
#     for dicom_file in dicom_files:
#         dicom_data = pydicom.dcmread(dicom_file)
#         if dicom_data.Modality != 'RTSTRUCT':
#             img_data[:, :, dicom_data.InstanceNumber - 1] = dicom_data.pixel_array

#     # Create an affine transformation matrix for the NIfTI file
#     if hasattr(dicom_data, 'affine'):
#         affine = dicom_data.affine
#     elif (
#         hasattr(dicom_data, 'ImageOrientationPatient') and 
#         hasattr(dicom_data, 'ImagePositionPatient') and 
#         hasattr(dicom_data, 'PixelSpacing')
#     ):
#         orientation = np.array(dicom_data.ImageOrientationPatient).reshape((2, 3))
#         position = np.array(dicom_data.ImagePositionPatient)
#         spacing = np.array(dicom_data.PixelSpacing)
#         direction = np.cross(orientation, orientation[1])
#         affine = np.eye(4)
#         affine[:3, :3] = orientation * spacing
#         affine[:3, 3] = position
#         affine[3, :3] = direction
#         affine = np.linalg.inv(affine)
#     else:
#         affine = np.eye(4)
#     # else:
#     #     affine = np.eye(4)
#     #     affine[:2, :2] = np.array(sample_dicom.PixelSpacing) * img_shape[:2]
#     #     affine[2, 2] = np.array(sample_dicom.SliceThickness)

#     # Create a NIfTI image from the 3D numpy array and save it to the output file
#     nifti_img = nib.Nifti1Image(img_data, affine)
#     nib.save(nifti_img, output_nifti_file)
    