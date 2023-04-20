import os
import pydicom
import numpy as np
import SimpleITK as sitk
def load_dicom_folder(folder_path):
    """Load a series of DICOM images in a folder (excluding RTStruct files).

    Args:
        folder_path (str): Path to the folder containing the DICOM files.

    Returns:
        dicom_files (list): List of file paths to the DICOM files, sorted by slice position.
        spacing (tuple): Tuple containing the voxel spacing of the images.
        origin (tuple): Tuple containing the origin of the images.
        direction (ndarray): 3x3 numpy array containing the direction cosines of the images.
        patient_id (str): Patient ID of the DICOM files.
    """
    # Get a list of all the DICOM files in the folder
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith('.dcm') and 'rtstruct' not in f.lower()]

    # Sort the DICOM files by slice position
    dicom_files = sorted(dicom_files, key=lambda x: pydicom.dcmread(x).ImagePositionPatient[2])

    # Get the voxel spacing, origin, direction, and patient ID from the first DICOM file
    ds = pydicom.dcmread(dicom_files[0])
    spacing = tuple(ds.PixelSpacing) + (float(ds.SliceThickness),)
    origin = tuple(ds.ImagePositionPatient)
    direction = np.array(ds.ImageOrientationPatient).reshape(2, 3)
    direction = np.vstack((direction, np.cross(direction[0], direction[1])))
    patient_id = ds.PatientID

    # Verify that the DICOM files belong to the same patient and have consistent metadata
    for file_path in dicom_files[1:]:
        ds = pydicom.dcmread(file_path)
        if ds.PatientID != patient_id:
            raise ValueError('DICOM files do not belong to the same patient.')
        if tuple(ds.PixelSpacing) + (float(ds.SliceThickness),) != spacing:
            raise ValueError('DICOM files have inconsistent voxel spacing or slice thickness.')
        if not np.allclose(np.array(ds.ImageOrientationPatient).reshape(2, 3), direction[:2, :3], rtol=1e-3):
            raise ValueError('DICOM files have inconsistent direction cosines.')

    return dicom_files #spacing, origin, direction, patient_id


def resample_image(image, spacing):
    """Resample an image to have the given voxel spacing.

    Args:
        image (SimpleITK.Image): The input image.
        spacing (tuple): The desired voxel spacing.

    Returns:
        resampled_image (SimpleITK.Image): The resampled image.
    """
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(spacing)
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.SetSize(image.GetSize())
    resample_filter.SetInterpolator(sitk.sitkLinear)
    resampled_image = resample_filter.Execute(image)
    return resampled_image

def calculate_affine(dicom_files, new_spacing=None):
    """Calculate the affine transformation matrix from a list of DICOM files.

    Args:
        dicom_files (list): List of file paths to the DICOM files.
        resample (bool): Whether to resample the image to have constant voxel spacing. Default is True.

    Returns:
        affine (ndarray): The affine transformation matrix.
"""
    # Load the DICOM files into a SimpleITK image
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    # Resample the image if requested or if the voxel spacing is not consistent
    spacing = np.array(image.GetSpacing())
    if new_spacing is not None or not np.allclose(spacing, spacing[0], rtol=1e-3):
        if new_spacing is None:
            new_spacing = (1.0, 1.0, 1.0)
            # Notify the user that the image will be resampled
            print('Image voxel spacing is not consistent or does not match the desired spacing. Resampling to (1, 1, 1) mm.')
        
        # Resample the image to have the desired voxel spacing
        resampled_image = resample_image(image, new_spacing)
        
        # Update the variables with the resampled image's data
        spacing = resampled_image.GetSpacing()
        origin = resampled_image.GetOrigin()
        direction = np.array(resampled_image.GetDirection()).reshape(3, 3)
    else:
        # Use the original image's data
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = np.array(image.GetDirection()).reshape(3, 3)

    # Create the affine transformation matrix
    affine = np.zeros((4, 4))
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    affine[0:3, 3] = origin
    affine[3, 3] = 1
    affine[0:3, 0:3] = direction

    return affine

def convert_to_nifti(image, output_file):
    """Convert a SimpleITK image to NIfTI format and save it to disk.
    Args:
        image (SimpleITK.Image): The input image.
        output_file (str): The path to the output NIfTI file.
    """
    sitk.WriteImage(image, output_file)
