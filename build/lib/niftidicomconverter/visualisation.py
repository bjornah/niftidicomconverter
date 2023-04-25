import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import nibabel as nib

def visualize_itk_slice(image: sitk.Image, slice_idx: int) -> None:
    """
    Display a 2D slice of an ITK image using matplotlib.

    Args:
        image (sitk.Image): The ITK image to visualize.
        slice_idx (int): The index of the slice to display.
    """
    # Get the numpy array representation of the image data
    image_np = sitk.GetArrayFromImage(image)
    
    # Get the 2D slice at the specified index
    slice_np = image_np[slice_idx, :, :]
    
    # Plot the slice using matplotlib
    plt.imshow(slice_np, cmap='gray')
    plt.axis('off')
    plt.show()
    
def visualize_nifti_slice(img: nib.nifti1.Nifti1Image, slice_index: int, cmap: str = 'gray'):
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

    # Get the slice to visualize
    if slice_index is not None:
        slice_data = data[slice_index, :, :]
    else:
        slice_data = data[:, :]

    fig, ax = plt.subplots()
    ax.imshow(slice_data.T, cmap=cmap) #, aspect=affine[0, 0]/affine[1, 1])

    # Set the axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Show the plot
    plt.show()