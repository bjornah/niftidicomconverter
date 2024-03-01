import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import nibabel as nib

from typing import Tuple, Optional, List, Union

from niftidicomconverter.dicomhandling import load_dicom_images
from niftidicomconverter.dicomhandling_pydicom import load_dicom_images_pydicom
from niftidicomconverter.niftihandling import read_nifti_file_sitk, read_nifti_file_nib

def visualize_itk_slice(image: sitk.Image, slice_index: int) -> None:
    """
    Display a 2D slice of an ITK image using matplotlib.

    Args:
        image (sitk.Image): The ITK image to visualize.
        slice_index (int): The index of the slice to display.
    """
    # Get the numpy array representation of the image data
    image_np = sitk.GetArrayFromImage(image)
    
    array = squeeze_if_one(image_np)
    
    # Get the 2D slice at the specified index
    if slice_index is not None and len(array.shape) > 2:
        slice_np = array[slice_index, :, :]
    else:
        slice_np = array
        
    # Plot the slice using matplotlib
    plt.imshow(slice_np, cmap='gray')
    plt.show()
    
def visualize_nifti_slice(img: nib.nifti1.Nifti1Image, slice_index: int, cmap: str = 'gray') -> None:
    """
    Visualizes a slice of a Nifti image using Matplotlib.

    Args:
        nifti_file (str): The path to the Nifti file.
        slice_index (int): The index of the slice to visualize.
        cmap (Optional[str]): The colormap to use. Default is 'gray'.
    """
    # Get the data and the affine transformation
    data = img.get_fdata()
    affine = img.affine

    array = squeeze_if_one(data)

    # Get the slice to visualize
    if slice_index is not None and len(array.shape) > 2:
        slice_data = array[slice_index, :, :]
    else:
        slice_data = array
    # else:
    #     slice_data = array[:, :]

    fig, ax = plt.subplots()
    ax.imshow(slice_data.T, cmap=cmap) #, aspect=affine[0, 0]/affine[1, 1])

    # Set the axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Show the plot
    plt.show()
    
def squeeze_if_one(arr: np.ndarray) -> np.ndarray:
    """
    Squeezes the input array along any dimension of size 1.

    Parameters:
        arr (np.ndarray): Input array to be squeezed.

    Returns:
        np.ndarray: Squeezed array if any dimension is of size 1, otherwise the original array.
    """
    if 1 in arr.shape:
        return np.squeeze(arr)
    else:
        return arr
    
def visualize_dicom_file(dicom_path: Union[str, List[str]], slice_index: int = 0, loader: str = 'sitk', ax=None, **kwargs) -> None:
    """
    Visualizes a DICOM file or series of files.

    Parameters:
        dicom_path (Union[str, List[str]]): Path to a DICOM file or directory containing DICOM files.
        slice_index (int): Index of the slice to display (default is 0).
        loader (str): DICOM loader to use, either 'sitk' or 'pydicom' (default is 'sitk').
        ax (matplotlib.axes.Axes): Matplotlib axes object to display the image on (default is None).
        **kwargs: Additional keyword arguments to pass to the DICOM loader.

    Returns:
        None: The function displays the image on the provided axes object.
    """
    if loader=='sitk':
        image_itk = load_dicom_images(dicom_path, **kwargs)
        array = sitk.GetArrayFromImage(image_itk)
    elif loader=='pydicom':
        pydicom_data = load_dicom_images_pydicom(dicom_path, **kwargs)
        array = pydicom_data.PixelData
        
    array = squeeze_if_one(array)

    if ax is None:
        fig, ax = plt.subplots()
    if len(array.shape)==2:
        ax.imshow(array, cmap='gray')
    else:
        ax.imshow(array[:,:,slice_index], cmap='gray')
        
def plot_dicom_matrix(nrows: int, ncols: int, dicom_files: List[str], figsize: Tuple[int, int] = (10,10), 
                      indices: Optional[List[int]] = None, loader: str = 'sitk', **kwargs) -> None:
    """
    Displays multiple DICOM files in a matrix of subplots.

    Parameters:
        nrows (int): Number of rows in the matrix.
        ncols (int): Number of columns in the matrix.
        dicom_files (List[str]): List of paths to DICOM files to display.
        figsize (Tuple[int, int]): Figure size (default is (10,10)).
        indices (List[int]): List of slice indices to display (default is None, which displays the first slice of each file).

    Returns:
        None: The function displays the image matrix on the screen.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for i, ax in enumerate(axes.ravel()):
        if i<len(dicom_files):
            visualize_dicom_file(dicom_files[i], ax=ax, loader=loader, **kwargs)
            
def visualize_nifti_file(nii_path: Union[str, List[str]], slice_index: int = 0, loader: str = 'sitk', 
                         ax=None, **kwargs) -> None:
    """
    Visualizes a single slice or 2D NIfTI file using SimpleITK or NiBabel for loading the image data.

    Args:
        nii_path (Union[str, List[str]]): Path to the NIfTI file(s) to be visualized. Can be a single string path or a list
            of paths for batch processing with SimpleITK only.
        slice_index (int, optional): The index of the slice to be visualized if the image is 3D. Defaults to 0.
        loader (str, optional): Specifies the library used for loading the NIfTI file. Valid values are 'sitk' for SimpleITK
            and 'nib' for NiBabel. Defaults to 'sitk'.
        ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to display the image. If None, a new figure and
            axis are created.
        **kwargs: Additional keyword arguments passed to the NIfTI file loading function.

    Returns:
        None: The function does not return a value but visualizes the NIfTI file directly.

    Example:
        >>> visualize_nifti_file('path/to/image.nii', slice_index=10, loader='nib')
    """

    if loader=='sitk':
        image_itk = read_nifti_file_sitk(nii_path, **kwargs)
        array = sitk.GetArrayFromImage(image_itk)
    elif loader=='nib':
        nib_file = read_nifti_file_nib(nii_path, **kwargs)
        array = nib_file.get_fdata()
        
    array = squeeze_if_one(array)

    if ax is None:
        fig, ax = plt.subplots()
    if len(array.shape)==2:
        ax.imshow(array, cmap='gray')
    else:
        ax.imshow(array[:,:,slice_index], cmap='gray')
        
def plot_nifti_matrix(nrows: int, ncols: int, nii_files: List[str], figsize=(10, 10), indices: Optional[List[int]] = None, loader: str = 'sitk', **kwargs) -> None:
    """
    Plots a matrix of NIfTI images, visualizing a single slice or 2D images from each file in a subplot grid.

    This function creates a grid of subplots with the specified number of rows and columns, then populates each subplot
    with an image from the list of NIfTI files provided. It uses `visualize_nifti_file` to handle the visualization of
    each file, supporting both SimpleITK and NiBabel for image loading. The slice index for 3D images can be specified
    through `indices`, allowing different slices to be visualized for each image.

    Args:
        nrows (int): Number of rows in the subplot grid.
        ncols (int): Number of columns in the subplot grid.
        nii_files (List[str]): List of paths to NIfTI files to be visualized.
        figsize ((int, int), optional): Figure size in inches. Defaults to (10, 10).
        indices (Optional[List[int]], optional): List of slice indices to visualize for each NIfTI file. If provided, must
            match the length of `nii_files`. If None, the first slice (index 0) is visualized for each file. Defaults to None.
        loader (str, optional): Specifies the library used for loading the NIfTI files. Valid values are 'sitk' for
            SimpleITK and 'nib' for NiBabel. Defaults to 'sitk'.
        **kwargs: Additional keyword arguments passed to the NIfTI file loading and visualization functions.

    Returns:
        None: The function does not return a value but plots the NIfTI images directly.

    Example:
        >>> nii_files = ['path/to/image1.nii', 'path/to/image2.nii']
        >>> plot_nifti_matrix(1, 2, nii_files, loader='nib')
    """


    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for i, ax in enumerate(axes.ravel()):
        if i<len(nii_files):
            visualize_nifti_file(nii_files[i], ax=ax, loader=loader, **kwargs)