import pytest
import SimpleITK as sitk
import numpy as np
from niftidicomconverter.dicomhandling import check_itk_image, itk_resample_volume, check_dicom_metadata_consistency, load_dicom_images, get_dicom_files, get_affine_from_itk_image, save_itk_image_as_nifti_nib, save_itk_image_as_nifti_sitk, permute_itk_axes

# You can replace 'path_to_dicom_folder' with a valid path containing DICOM files for testing purposes
# path_to_dicom_folder = 'data/brainarteries'
path_to_dicom_folder = 'data/dicom_rsna_files_id5'

def test_check_itk_image():
    # Create a simple 3D image with uniform spacing
    img = sitk.GaussianSource(size=[10, 10, 10], sigma=[1.0, 1.0, 1.0], mean=[5.0, 5.0, 5.0])
    assert check_itk_image(img) == True

def test_itk_resample_volume():
    # Create a simple 3D image with uniform spacing
    img = sitk.GaussianSource(size=[10, 10, 10], sigma=[1.0, 1.0, 1.0], mean=[5.0, 5.0, 5.0])
    resampled_img = itk_resample_volume(img, new_spacing=(0.5, 0.5, 0.5))
    resampled_spacing = resampled_img.GetSpacing()
    assert np.allclose(resampled_spacing, (0.5, 0.5, 0.5))

def test_check_dicom_metadata_consistency():
    dicom_files = get_dicom_files(path_to_dicom_folder)
    assert check_dicom_metadata_consistency(dicom_files) == True

def test_load_dicom_images():
    image = load_dicom_images(dicom_path=path_to_dicom_folder)
    assert isinstance(image, sitk.Image)

def test_get_dicom_files():
    dicom_files = get_dicom_files(path_to_dicom_folder)
    assert len(dicom_files) > 0

def test_get_affine_from_itk_image():
    img = sitk.GaussianSource(size=[10, 10, 10], sigma=[1.0, 1.0, 1.0], mean=[5.0, 5.0, 5.0])
    affine = get_affine_from_itk_image(img)
    assert affine.shape == (4, 4)

def test_save_itk_image_as_nifti_nib():
    img = sitk.GaussianSource(size=[10, 10, 10], sigma=[1.0, 1.0, 1.0], mean=[5.0, 5.0, 5.0])
    save_itk_image_as_nifti_nib(img, 'test_nifti_nib.nii.gz')

def test_save_itk_image_as_nifti_sitk():
    img = sitk.GaussianSource(size=[10, 10, 10], sigma=[1.0, 1.0, 1.0], mean=[5.0, 5.0, 5.0])
    save_itk_image_as_nifti_sitk(img, 'test_nifti_sitk.nii.gz')

def test_permute_itk_axes():
    img = sitk.GaussianSource(size=[10, 10, 10], sigma=[1.0, 1.0, 1.0], mean=[5.0, 5.0, 5.0])
    permuted_img = permute_itk_axes(img, permute_axes=(2, 1, 0))
    assert img.GetSize() == tuple(reversed(permuted_img.GetSize()))

