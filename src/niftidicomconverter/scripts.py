import os
import logging
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
import tqdm
from pydicom import dcmread
from niftidicomconverter.DicomToNifti import convert_dicom_series_to_nifti_image, convert_dicom_rtss_to_nifti
from niftidicomconverter.utils import correct_nifti_header, generate_dicom_filename
from niftidicomconverter.dicomhandling import pair_dicom_series_v2
from niftidicomconverter.niftihandling import resample_nifti_to_new_spacing_sitk

def batch_convert_dicom_to_nifti(base_path, output_base_dir, structure_map, df_kwargs={}, new_spacing=None):
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
        log_file (str): The path to the log file.
        **df_kwargs: Additional columns to include in the metadata. 'site' must always be included.
    """

    # Ensure 'site' is always included in df_kwargs
    if 'site' not in df_kwargs:
        df_kwargs['site'] = None

    res_list = pair_dicom_series_v2(base_path)
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    metadata_list = []
    csv_file_path = os.path.join(output_base_dir, 'database_file.csv')

    # Check if the CSV file exists and delete it if it does
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
        logging.info(f'Deleted existing CSV file: {csv_file_path}')

    for file_dict in tqdm.tqdm(res_list, desc='Converting DICOM to Nifti'):
        dicom_image_files, rtss = file_dict['image'], file_dict['rtss']
        logging.info(f'dicom file paths = {dicom_image_files[0]}')
        logging.info(f'rtss file = {rtss}')
        
        if dicom_image_files:
            try:
                dicom_file_path = dicom_image_files[0]
                fname = generate_dicom_filename(dicom_file_path)

                output_dir = os.path.join(output_base_dir, fname)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                output_fname = os.path.join(output_dir, fname + '.nii.gz')
                convert_dicom_series_to_nifti_image(dicom_image_files, output_fname)
                
                # Extract DICOM metadata
                dicom_metadata = dcmread(dicom_file_path)
                patient_id = getattr(dicom_metadata, 'PatientID', None)
                study_date = getattr(dicom_metadata, 'StudyDate', None)
                modality = getattr(dicom_metadata, 'Modality', None)
                sop_class_uid = getattr(dicom_metadata, 'SOPClassUID', None)
                original_resolution = [float(x) for x in getattr(dicom_metadata, 'PixelSpacing', [None, None])] + [float(getattr(dicom_metadata, 'SliceThickness', None))]
                study_instance_uid = getattr(dicom_metadata, 'StudyInstanceUID', None)
                series_instance_uid = getattr(dicom_metadata, 'SeriesInstanceUID', None)
                series_description = getattr(dicom_metadata, 'SeriesDescription', None)

                nifti_image = nib.load(output_fname)
                original_size = nifti_image.shape
                resulting_resolution = [float(x) for x in nifti_image.header.get_zooms()]
                resulting_size = nifti_image.shape

                rtss_sop_instance_uid = None
                if rtss:
                    rtss_metadata = dcmread(rtss)
                    rtss_sop_instance_uid = getattr(rtss_metadata, 'SOPInstanceUID', None)

            except Exception as e:
                logging.exception(f'Error converting DICOM image to NIfTI: {e}')
                logging.error(f'dicom_image_files = {dicom_image_files}')
                
            if rtss:
                try:
                    output_fname_rtss = os.path.join(output_dir, fname + '_RTSS.nii.gz')
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
                    nifti_image = nib.load(output_fname)
                    resulting_resolution = [float(x) for x in nifti_image.header.get_zooms()]
                    resulting_size = nifti_image.shape
                except Exception as e:
                    logging.exception(f'Error resampling NIfTI files: {e}')
                    logging.error(f'output_fname = {output_fname}')
                    logging.error(f'output_fname_rtss = {output_fname_rtss}')
            
            # Collect metadata
            metadata = {
                'PatientID': patient_id,
                'StudyDate': study_date,
                'Modality': modality,
                'NiftiFilePath': output_fname,
                'NiftiRTSSFilePath': output_fname_rtss if rtss else None,
                'DicomStackPath': dicom_image_files,
                'DicomRTSSPath': rtss,
                'SOPClassUID': sop_class_uid,
                'OriginalResolution': original_resolution,
                'ResultingResolution': resulting_resolution,
                'OriginalSize': original_size,
                'ResultingSize': resulting_size,
                'StudyInstanceUID': study_instance_uid,
                'SeriesInstanceUID': series_instance_uid,
                'RTSS_SOPInstanceUID': rtss_sop_instance_uid,
                'SeriesDescription': series_description,
            }
            # Add additional columns from df_kwargs
            metadata.update(df_kwargs)
            metadata_list.append(metadata)
            
            # Save metadata incrementally
            df = pd.DataFrame([metadata])
            if not os.path.isfile(csv_file_path):
                df.to_csv(csv_file_path, index=False)
            else:
                df.to_csv(csv_file_path, mode='a', header=False, index=False)
    
    # Create DataFrame from all metadata
    df = pd.DataFrame(metadata_list)
    return df

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

    logging.debug('after resampling')
    nifti_image_tmp = nib.load(nifti_image_output)
    logging.debug(f'nifti_image.shape = {nifti_image_tmp.get_fdata().shape}')
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

        logging.debug('after resampling')
        nifti_rtss_tmp = nib.load(nifti_rtss_output)
        logging.debug(f'nifti_gt.shape = {nifti_rtss_tmp.get_fdata().shape}')
        logging.debug(f'nifti_gt.affine = {nifti_rtss_tmp.affine}')