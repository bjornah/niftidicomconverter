import tqdm
import os
import logging
import nibabel as nib
import SimpleITK as sitk
from niftidicomconverter.DicomToNifti import convert_dicom_series_to_nifti_image, convert_dicom_rtss_to_nifti
from niftidicomconverter.utils import correct_nifti_header, generate_dicom_filename
from niftidicomconverter.dicomhandling import pair_dicom_series_v2
from niftidicomconverter.niftihandling import resample_nifti_to_new_spacing_sitk

def batch_convert_dicom_to_nifti(base_path, output_base_dir, structure_map, new_spacing=None):
    """
    Batch converts DICOM files to NIfTI format. Searches through base_path recursively and identifies
    DICOM series. Converts the DICOM series to NIfTI format and saves them in the output_base_dir, along 
    with the corresponding RTSS files. The structure_map is used to map the RTSS structures to the
    channels in the NIfTI files. If structure_map is None, all structures are included in the nifti file.

    Args:
        base_path (str): The base path where the DICOM files are located.
        output_base_dir (str): The base directory where the converted NIfTI files will be saved.
        structure_map (dict): A dictionary mapping structure names to their corresponding values.
        new_spacing (Optional[Tuple[float, float, float]]): The desired voxel spacing for the resampled nifti files.

    Returns:
        None
    """
    res_list = pair_dicom_series_v2(base_path)
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    for file_dict in tqdm.tqdm(res_list, desc='Converting DICOM to Nifti'):
        dicom_image_files, rtss = file_dict['image'], file_dict['rtss']
        if dicom_image_files:
            try:
                dicom_file_path = dicom_image_files[0]
                fname = generate_dicom_filename(dicom_file_path)

                output_dir = os.path.join(output_base_dir, fname)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                output_fname = os.path.join(output_dir, fname + '.nii.gz')
                convert_dicom_series_to_nifti_image(dicom_image_files, output_fname)
            except Exception as e:
                logging.exception(f'Error converting DICOM image to NIfTI: {e}')
                logging.error(f'dicom_image_files = {dicom_image_files}')
                
            if rtss:
                try:
                    output_fname_rtss = os.path.join(output_dir, fname + '_RTSS.nii.gz')
                    # import pdb; pdb.set_trace()
                    convert_dicom_rtss_to_nifti(dicom_image_files, rtss, output_fname_rtss, structure_map=structure_map)
                    
                    # fix nifti rtss headers and affine matrix
                    src = output_fname
                    dst = output_fname_rtss
                    new_nifti_rtss = correct_nifti_header(src, dst)
                    nib.save(new_nifti_rtss, dst)
                except Exception as e:
                    logging.exception(f'Error converting RTSS to NIfTI: {e}')
                    logging.error(f'dicom_image_files = {dicom_image_files}')
                    logging.error(f'rtss = {rtss}')
                    
        # resample to new resolution
        if new_spacing:
            try:
                resize_nifti_files(
                    nifti_image=output_fname,
                    nifti_image_output=output_fname,
                    nifti_rtss=output_fname_rtss,
                    nifti_rtss_output=output_fname_rtss,
                    new_spacing=new_spacing
                )
            except Exception as e:
                logging.exception(f'Error resampling NIfTI files: {e}')
                logging.error(f'output_fname = {output_fname}')
                logging.error(f'output_fname_rtss = {output_fname_rtss}')
                    
                    
def resize_nifti_files(nifti_image, nifti_image_output, nifti_rtss=None, nifti_rtss_output=None, new_spacing=(1, 1, 1)):
    """
    Resizes nifti images to a given spatial resolution. If an RTSS is also given, that is resampled as well to the same resolution.
    
    input parameters:
    nifti_image: str, path to the nifti image
    nifti_rtss: str, path to the nifti rtss file
    new_spacing: tuple, the new spatial resolution to resample the images to
    """

    logging.debug('before resampling')
    nifti_image_tmp = nib.load(nifti_image)
    # assert isinstance(nifti_image, nib.nifti1.Nifti1Image)
    
    logging.debug(f'nifti_image.shape = {nifti_image_tmp.get_fdata().shape}')
    logging.debug(f'nifti_image.affine = {nifti_image_tmp.affine}')

    # resample nifti to new spacing
    resample_nifti_to_new_spacing_sitk(
        nifti_file_path=nifti_image,
        new_spacing=new_spacing,
        save_path=nifti_image_output,
        interpolator=sitk.sitkLinear
    ) 

    # this reorients the nifti image to the standard dicom-style orientation. However, it deteriorates the quality of the predictions
    # nifti_nib = reorient_nifti(input_data=NIFTI_IMAGE, target_orientation=('A', 'R', 'S'), print_debug=True)
    # nib.save(nifti_nib, NIFTI_IMAGE)

    logging.debug('after resampling')
    nifti_image_tmp = nib.load(nifti_image_output)
    logging.debug(f'nifti_image.shape = {nifti_image_tmp.get_fdata().shape}')
    # print out the affine matrix of the nifti image
    logging.debug(f'nifti_image.affine = {nifti_image_tmp.affine}')

    if nifti_rtss is not None:
        assert nifti_rtss_output is not None, 'If an RTSS file is given, a save path must also be given for the RTSS file'
        
        logging.debug('before resampling')
        nifti_rtss_tmp = nib.load(nifti_rtss)
        logging.debug(f'nifti_gt.shape = {nifti_rtss_tmp.get_fdata().shape}')
        logging.debug(f'nifti_gt.affine = {nifti_rtss_tmp.affine}')

        # resample nifti to new spacing
        resample_nifti_to_new_spacing_sitk(
            nifti_file_path=nifti_rtss,
            new_spacing=new_spacing,
            save_path=nifti_rtss_output,
            interpolator=sitk.sitkNearestNeighbor
        ) 

        # this reorients the labels to the standard dicom-style orientation. This should only be done if you also transpose the image. However, it deteriorates the quality of the predictions
        # nifti_nib = reorient_nifti(input_data=NIFTI_GT, target_orientation=('A', 'R', 'S'), print_debug=True)
        # nib.save(nifti_nib, NIFTI_GT)

        logging.debug('after resampling')
        nifti_rtss_tmp = nib.load(nifti_rtss_output)
        logging.debug(f'nifti_gt.shape = {nifti_rtss_tmp.get_fdata().shape}')
        logging.debug(f'nifti_gt.affine = {nifti_rtss_tmp.affine}')