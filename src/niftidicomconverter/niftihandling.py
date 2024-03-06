import SimpleITK as sitk
import nibabel as nib
from typing import Union
import numpy as np
from scipy.ndimage import zoom

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

def reorient_nifti(input_data: Union[str, nib.Nifti1Image], target_orientation: tuple, print_debug: bool=False) -> nib.Nifti1Image:
    """
    Reorient a NIFTI image to a specified target orientation.

    The target orientation is specified as a tuple of axis codes, representing each axis's direction in 3D space.
    These codes are:
    - 'R' or 'L' for the Right or Left side of the patient.
    - 'A' or 'P' for Anterior (front) or Posterior (back) of the patient.
    - 'S' or 'I' for Superior (top) or Inferior (bottom) of the patient.

    For example, ('R', 'A', 'S') means the first axis is Right-to-Left, the second is Anterior-to-Posterior, 
    and the third is Superior-to-Inferior.

    Parameters:
    input_data (Union[str, nib.Nifti1Image]): Path to the input NIFTI file or a loaded NIFTI image.
    target_orientation (tuple): Target orientation as a tuple of axis codes (e.g., ('R', 'A', 'S')).

    Returns:
    nib.Nifti1Image: A NIFTI image reoriented to the target orientation.

    Example:
    >>> reoriented_nifti = reorient_nifti("path/to/nifti/file.nii", ('R', 'A', 'S'))
    or
    >>> nifti_img = nib.load("path/to/nifti/file.nii")
    >>> reoriented_nifti = reorient_nifti(nifti_img, ('R', 'A', 'S'))
    """
    # Load the NIFTI file if input_data is a file path
    if isinstance(input_data, str):
        nifti_img = nib.load(input_data)
    else:
        nifti_img = input_data

    old_affine = nifti_img.affine
    
    old_shape = nifti_img.shape

    # Get current orientation and target orientation array
    old_ornt = nib.orientations.io_orientation(nifti_img.affine)
    old_axcode = nib.orientations.ornt2axcodes(old_ornt)

    target_ornt = nib.orientations.axcodes2ornt(target_orientation)

    # Find transformation from current to target orientation
    ornt_transformation = nib.orientations.ornt_transform(old_ornt, target_ornt)

    # Apply the transformation
    reoriented_data = nib.orientations.apply_orientation(nifti_img.get_fdata(), ornt_transformation)

    # Calculate the inverse orientation affine
    inv_affine = nib.orientations.inv_ornt_aff(ornt_transformation, old_shape)

    # Combine with the original affine
    new_affine = np.dot(old_affine, inv_affine)

    reoriented_nifti = nib.Nifti1Image(reoriented_data, new_affine)
    # new_ornt = nib.orientations.io_orientation(reoriented_nifti.affine)
    # print(f'new ornt = {new_ornt}')
    if print_debug:
        print(f'old affine = {old_affine}')
        print(f'old shape = {old_shape}')
        print(f'old ornt = {old_ornt}')
        print(f'old axcode = {old_axcode}')
        print(f'target ornt = {target_ornt}')
        print(f'ornt transformation = {ornt_transformation}')
        print(f'new affine = {new_affine}')
        print(f'new axcode = {nib.aff2axcodes(reoriented_nifti.affine)}')

    return reoriented_nifti

def resample_3d_volume(data, scale_factors, order=2):
    """
    Resample a single 3D volume.

    Args:
        data: The 3D numpy array to be resampled.
        scale_factors: The scale factors for resampling.
        order: The order of the spline interpolation. Default is 2 (bi-linear).

    Returns:
        The resampled 3D volume as a numpy array.
    """
    return zoom(data, scale_factors, order=order)

def resample_nifti_to_new_spacing(nifti_file_path: str, new_spacing: tuple[float, float, float], save_path: str = None) -> nib.Nifti1Image:
    """
    Resample a NIfTI image (3D or 4D) to the specified spacing for the first three dimensions, preserving the fourth dimension.

    Args:
        nifti_file_path: The file path to the input NIfTI image.
        new_spacing: A tuple of three floats representing the new spacing (x, y, z).
        save_path: Optional. If provided, the resampled NIfTI image will be saved to this path.

    Returns:
        The resampled NIfTI image as a nibabel Nifti1Image object.

    Note:
        This function does not interpolate across the fourth dimension if present.
    """
    # Load the NIfTI file
    nifti_img = nib.load(nifti_file_path)
    data = nifti_img.get_fdata()
    original_affine = nifti_img.affine

    # Calculate the scale factors for each dimension
    original_spacing = np.sqrt((original_affine[:3, :3] ** 2).sum(axis=0))
    scale_factors = original_spacing / np.array(new_spacing)

    # Check if the image is 4D
    if data.ndim == 4:
        # Resample each 3D volume independently
        resampled_data = np.array([resample_3d_volume(data[:, :, :, i], scale_factors, order=2) for i in range(data.shape[3])]).transpose((1, 2, 3, 0))
    else:
        # Resample the 3D image
        resampled_data = resample_3d_volume(data, scale_factors, order=2)

    # Construct the new affine matrix
    new_affine = original_affine.copy()
    np.fill_diagonal(new_affine, np.append(new_spacing, [1]))
    new_affine[:3, 3] = original_affine[:3, 3] * scale_factors  # Adjust translation part

    # Create a new NIfTI image with the resampled data and new affine
    resampled_nifti_img = nib.Nifti1Image(resampled_data, new_affine, header=nifti_img.header)

    # Update the header (e.g., update the pixdim to reflect the new spacing)
    resampled_nifti_img.header['pixdim'][1:4] = new_spacing

    # Save the resampled image if a save path is provided
    if save_path:
        nib.save(resampled_nifti_img, save_path)

    return resampled_nifti_img


# def resample_nifti_rtss(
#     nifti_rtss_path:str,
#     output_nifti_path:str,
#     new_spacing: Optional[Tuple[float, float, float]] = None, 
# ):
#     rtss_sitk = read_nifti_file_sitk(nifti_rtss_path)

#     print(f'resampling image to resolution {new_spacing} mm')
#     image = itk_resample_volume(img=image, new_spacing=new_spacing)

#     # this is to get the header of the original dicom
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tmpfile = os.path.join(tmp_dir, 'image.nii')

#         dicom_image_sitk = load_dicom_images(dicom_folder)
#         save_itk_image_as_nifti_sitk(dicom_image_sitk, tmpfile)

#         nifti_image_src = nib.load(tmpfile)

#         nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)

#     nib.save(nib_rtss, output_nifti_path)