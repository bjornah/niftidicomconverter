import os
import logging
import glob

import nibabel as nib
import SimpleITK as sitk
import pandas as pd

from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple
from pydicom import dcmread
from niftidicomconverter.DicomToNifti import convert_dicom_series_to_nifti_image, convert_dicom_rtss_to_nifti
from niftidicomconverter.utils import correct_nifti_header, generate_dicom_filename
from niftidicomconverter.dicomhandling import pair_dicom_series_v5
from niftidicomconverter.niftihandling import resample_nifti_to_new_spacing_sitk
from niftidicomconverter.dicomhandling import get_subfolders_at_level


# def batch_convert_dicom_to_nifti(base_path, output_base_dir, structure_map, df_kwargs={}, new_spacing=None, glob_pattern=None):
#     """
#     Batch converts DICOM files to NIfTI format. Searches through base_path recursively and identifies
#     DICOM series. Converts the DICOM series to NIfTI format and saves them in the output_base_dir, along 
#     with the corresponding RTSS files. The structure_map is used to map the RTSS structures to the
#     channels in the NIfTI files. If structure_map is None, all structures are included in the nifti file.

#     Args:
#         base_path (str): The base path where the DICOM files are located.
#         output_base_dir (str): The base directory where the converted NIfTI files will be saved.
#         structure_map (dict): A dictionary mapping structure names to their corresponding values.
#         new_spacing (Optional[Tuple[float, float, float]]): The desired voxel spacing for the resampled nifti files.
#         log_file (str): The path to the log file.
#         **df_kwargs: Additional columns to include in the metadata. 'site' must always be included.
#     """

#     # Ensure 'site' is always included in df_kwargs
#     if 'site' not in df_kwargs:
#         df_kwargs['site'] = None

#     res_list = pair_dicom_series_v5(base_path, glob_pattern=glob_pattern)
    
#     if not os.path.exists(output_base_dir):
#         os.makedirs(output_base_dir)
    
#     metadata_list = []
#     csv_file_path = os.path.join(output_base_dir, 'database_file.csv')

#     # Check if the CSV file exists and delete it if it does
#     if os.path.exists(csv_file_path):
#         os.remove(csv_file_path)
#         logging.info(f'Deleted existing CSV file: {csv_file_path}')

#     for file_dict in tqdm.tqdm(res_list, desc='Converting DICOM to Nifti'):
#         dicom_image_files, rtss = file_dict['image'], file_dict['rtss']
#         logging.info(f'dicom file paths = {dicom_image_files[0]}')
#         logging.info(f'rtss file = {rtss}')
        
#         if dicom_image_files:
#             try:
#                 dicom_file_path = dicom_image_files[0]
#                 fname = generate_dicom_filename(dicom_file_path)

#                 output_dir = os.path.join(output_base_dir, fname)
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
                
#                 output_fname = os.path.join(output_dir, fname + '.nii.gz')
#                 convert_dicom_series_to_nifti_image(dicom_image_files, output_fname)
                
#                 # Extract DICOM metadata
#                 dicom_metadata = dcmread(dicom_file_path)
#                 patient_id = getattr(dicom_metadata, 'PatientID', None)
#                 study_date = getattr(dicom_metadata, 'StudyDate', None)
#                 modality = getattr(dicom_metadata, 'Modality', None)
#                 sop_class_uid = getattr(dicom_metadata, 'SOPClassUID', None)
#                 original_resolution = [float(x) for x in getattr(dicom_metadata, 'PixelSpacing', [None, None])] + [float(getattr(dicom_metadata, 'SliceThickness', None))]
#                 study_instance_uid = getattr(dicom_metadata, 'StudyInstanceUID', None)
#                 series_instance_uid = getattr(dicom_metadata, 'SeriesInstanceUID', None)
#                 series_description = getattr(dicom_metadata, 'SeriesDescription', None)

#                 nifti_image = nib.load(output_fname)
#                 original_size = nifti_image.shape
#                 resulting_resolution = [float(x) for x in nifti_image.header.get_zooms()]
#                 resulting_size = nifti_image.shape

#                 rtss_sop_instance_uid = None
#                 if rtss:
#                     rtss_metadata = dcmread(rtss)
#                     rtss_sop_instance_uid = getattr(rtss_metadata, 'SOPInstanceUID', None)

#             except Exception as e:
#                 logging.exception(f'Error converting DICOM image to NIfTI: {e}')
#                 logging.error(f'dicom_image_files = {dicom_image_files}')
                
#             if rtss:
#                 try:
#                     output_fname_rtss = os.path.join(output_dir, fname + '_RTSS.nii.gz')
#                     convert_dicom_rtss_to_nifti(dicom_image_files, rtss, output_fname_rtss, structure_map=structure_map)
                    
#                     # fix nifti rtss headers and affine matrix
#                     src = output_fname
#                     dst = output_fname_rtss
#                     new_nifti_rtss = correct_nifti_header(src, dst)
#                     nib.save(new_nifti_rtss, dst)
#                 except Exception as e:
#                     logging.exception(f'Error converting RTSS to NIfTI: {e}')
#                     logging.error(f'dicom_image_files = {dicom_image_files}')
#                     logging.error(f'rtss = {rtss}')
                    
#             # resample to new resolution
#             if new_spacing:
#                 try:
#                     resize_nifti_files(
#                         nifti_image=output_fname,
#                         nifti_image_output=output_fname,
#                         nifti_rtss=output_fname_rtss,
#                         nifti_rtss_output=output_fname_rtss,
#                         new_spacing=new_spacing
#                     )
#                     nifti_image = nib.load(output_fname)
#                     resulting_resolution = [float(x) for x in nifti_image.header.get_zooms()]
#                     resulting_size = nifti_image.shape
#                 except Exception as e:
#                     logging.exception(f'Error resampling NIfTI files: {e}')
#                     logging.error(f'output_fname = {output_fname}')
#                     logging.error(f'output_fname_rtss = {output_fname_rtss}')
            
#             dicom_image_base_path = os.path.dirname(dicom_image_files[0])

#             # Collect metadata
#             metadata = {
#                 'PatientID': patient_id,
#                 'StudyDate': study_date,
#                 'Modality': modality,
#                 'nii_file': output_fname,
#                 'RTSS_nii_file': output_fname_rtss if rtss else None,
#                 'DicomStackPath': dicom_image_base_path,
#                 'DicomRTSSPath': rtss,
#                 'SOPClassUID': sop_class_uid,
#                 'OriginalResolution': original_resolution,
#                 'ResultingResolution': resulting_resolution,
#                 'OriginalSize': original_size,
#                 'ResultingSize': resulting_size,
#                 'StudyInstanceUID': study_instance_uid,
#                 'SeriesInstanceUID': series_instance_uid,
#                 'RTSS_SOPInstanceUID': rtss_sop_instance_uid,
#                 'SeriesDescription': series_description,
#             }
#             # Add additional columns from df_kwargs
#             metadata.update(df_kwargs)
#             metadata_list.append(metadata)
            
#             # Save metadata incrementally
#             df = pd.DataFrame([metadata])
#             if not os.path.isfile(csv_file_path):
#                 df.to_csv(csv_file_path, index=False)
#             else:
#                 df.to_csv(csv_file_path, mode='a', header=False, index=False)
    
#     # Create DataFrame from all metadata
#     df = pd.DataFrame(metadata_list)
#     return df

# def batch_convert_dicom_to_nifti(
#     base_path: str,
#     output_base_dir: str,
#     structure_map: Optional[dict],
#     df_kwargs: dict = {},
#     new_spacing: Optional[Tuple[float, float, float]] = None,
#     folder_pattern: str = '*',
#     file_pattern: str = '*dcm',
#     incremental: bool = True,
#     subdir_level: int = 0,
#     metadata_csv_path: Optional[str] = None,
#     overwrite: bool = True
# ):
#     """
#     Batch converts DICOM files to NIfTI format. Searches through base_path recursively and identifies
#     DICOM series. Converts the DICOM series to NIfTI format and saves them in the output_base_dir, along 
#     with the corresponding RTSS files. The structure_map is used to map the RTSS structures to the
#     channels in the NIfTI files. If structure_map is None, all structures are included in the nifti file.

#     Args:
#         base_path (str): The base path where the DICOM files are located.
#         output_base_dir (str): The base directory where the converted NIfTI files will be saved.
#         structure_map (dict): A dictionary mapping structure names to their corresponding values.
#         new_spacing (Optional[Tuple[float, float, float]]): The desired voxel spacing for the resampled nifti files.
#         glob_pattern (Optional[str]): A glob pattern to filter files.
#         convert_by_subfolder_level (bool): If > 0, convert DICOM files in subfolders recursively.
#         metadata_csv_path (Optional[str]): Path to the CSV file containing metadata. If None, defaults to output directory.
#         overwrite (bool): If False, skip processing for files that have already been converted.
#         **df_kwargs: Additional columns to include in the metadata. 'site' must always be included.
#     """
#     # Ensure 'site' is always included in df_kwargs
#     if 'site' not in df_kwargs:
#         df_kwargs['site'] = None

#     # Load existing metadata if CSV exists
#     if metadata_csv_path is None:
#         metadata_csv_path = os.path.join(output_base_dir, 'database_file.csv')
#     if os.path.exists(metadata_csv_path) and not overwrite:
#         existing_database_df = pd.read_csv(metadata_csv_path)
#     else:
#         existing_database_df = pd.DataFrame()

#     top_level_path = Path(base_path)
#     grouped_files = None

#     if incremental:
#         # Get subfolders at the specified batch level using a generator
#         subfolders = (
#             sf for sf in get_subfolders_at_level(top_level_path, subdir_level)
#             if sf.match(folder_pattern)
#         )

#         # Add progress bar to the subfolders loop
#         for subfolder in tqdm(subfolders, desc='Processing Batches', unit='subfolder'):
#             # Collect files matching the file pattern within the subfolder
#             files_to_process = list(subfolder.glob('**/' + file_pattern))
#             if files_to_process:
#                 print(f"\nProcessing batch in {subfolder}")
#                 # Optionally, add a progress bar for collecting files
#                 files_to_process = list(tqdm(
#                     files_to_process,
#                     desc=f'Collecting files in {subfolder}',
#                     unit='file'
#                 ))
#                 paired_series = pair_dicom_series_v5(
#                     dicom_files=files_to_process,
#                     desired_sop_class_uid="1.2.840.10008.5.1.4.1.1.4",
#                     existing_database_df=existing_database_df
#                 )
#                 if len(paired_series) > 0:
#                     perform_conversion(
#                             paired_series,
#                             output_base_dir,
#                             existing_database_df,
#                             metadata_csv_path,
#                             structure_map,
#                             df_kwargs,
#                             new_spacing,
#                             overwrite,
#                         )
#                     overwrite = False  # Ensure subsequent writes append to the file

#     else:
#         # Use a generator to process folders as we find them
#         files_to_process = []
#         matching_folders = (
#             p for p in top_level_path.glob('**/' + folder_pattern) if p.is_dir()
#         )

#         # Use tqdm to add a progress bar
#         for folder in tqdm(matching_folders, desc='Scanning Folders', unit='folder'):
#             # Collect files matching the file_pattern within the folder
#             files_in_folder = list(folder.glob('**/' + file_pattern))
#             files_to_process.extend(files_in_folder)

#         if files_to_process:
#             print(f"\Found {len(files_to_process)} files in folders matching '{folder_pattern}' in {base_path}")
#             paired_series = pair_dicom_series_v5(
#                 dicom_files=files_to_process,
#                 desired_sop_class_uid="1.2.840.10008.5.1.4.1.1.4",
#                 existing_database_df=existing_database_df
#             )
#             if len(paired_series) > 0:
#                 perform_conversion(
#                         paired_series,
#                         output_base_dir,
#                         existing_database_df,
#                         metadata_csv_path,
#                         structure_map,
#                         df_kwargs,
#                         new_spacing,
#                         overwrite,
#                     )
#         else:
#             print("No files found to process.")

# def perform_conversion(
#         paired_series: str,
#         output_base_dir: str,
#         existing_database_df: pd.DataFrame,
#         metadata_csv_path: Optional[str],
#         structure_map: Optional[dict],
#         df_kwargs: dict = {},
#         new_spacing: Optional[Tuple[float, float, float]] = None,
#         overwrite: bool = True
#     ):
#     """
#     """
#     metadata_list = []

#     for file_dict in tqdm(paired_series, desc='Converting DICOM to NIfTI'):
#         dicom_image_files, rtss = file_dict['image'], file_dict['rtss']
#         logging.info(f'dicom file paths = {dicom_image_files[0]}')
#         logging.info(f'rtss file = {rtss}')

#         if dicom_image_files:
#             dicom_file_path = dicom_image_files[0]
#             fname = generate_dicom_filename(dicom_file_path)
#             output_dir = os.path.join(output_base_dir, fname)
#             output_fname = os.path.join(output_dir, fname + '.nii.gz')

#             # Skip processing if this series is already in the metadata CSV
#             if not overwrite and not existing_database_df.empty and existing_database_df['nii_file'].str.contains(output_fname).any():
#                 logging.info(f'Skipping already converted series: {output_fname}')
#                 continue

#             try:
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)

#                 convert_dicom_series_to_nifti_image(dicom_image_files, output_fname)

#                 # Extract DICOM metadata
#                 dicom_metadata = dcmread(dicom_file_path)
#                 patient_id = getattr(dicom_metadata, 'PatientID', None)
#                 study_date = getattr(dicom_metadata, 'StudyDate', None)
#                 modality = getattr(dicom_metadata, 'Modality', None)
#                 sop_class_uid = getattr(dicom_metadata, 'SOPClassUID', None)
#                 original_resolution = [float(x) for x in getattr(dicom_metadata, 'PixelSpacing', [None, None])] + [float(getattr(dicom_metadata, 'SliceThickness', None))]
#                 study_instance_uid = getattr(dicom_metadata, 'StudyInstanceUID', None)
#                 series_instance_uid = getattr(dicom_metadata, 'SeriesInstanceUID', None)
#                 series_description = getattr(dicom_metadata, 'SeriesDescription', None)
#                 protocol_name = getattr(dicom_metadata, 'ProtocolName', None)  # Extract ProtocolName
#                 sequence_name = getattr(dicom_metadata, 'SequenceName', None)  # Extract SequenceName
#                 scanning_sequence = getattr(dicom_metadata, 'ScanningSequence', None)  # Extract ScanningSequence

#                 nifti_image = nib.load(output_fname)
#                 original_size = nifti_image.shape
#                 resulting_resolution = [float(x) for x in nifti_image.header.get_zooms()]
#                 resulting_size = nifti_image.shape

#                 rtss_sop_instance_uid = None
#                 if rtss:
#                     rtss_metadata = dcmread(rtss)
#                     rtss_sop_instance_uid = getattr(rtss_metadata, 'SOPInstanceUID', None)

#             except Exception as e:
#                 logging.exception(f'Error converting DICOM image to NIfTI: {e}')
#                 logging.error(f'dicom_image_files = {dicom_image_files}')
#                 continue

#             if rtss:
#                 try:
#                     output_fname_rtss = os.path.join(output_dir, fname + '_RTSS.nii.gz')
#                     convert_dicom_rtss_to_nifti(dicom_image_files, rtss, output_fname_rtss, structure_map=structure_map)

#                     # Fix nifti rtss headers and affine matrix
#                     src = output_fname
#                     dst = output_fname_rtss
#                     new_nifti_rtss = correct_nifti_header(src, dst)
#                     nib.save(new_nifti_rtss, dst)
#                 except Exception as e:
#                     logging.exception(f'Error converting RTSS to NIfTI: {e}')
#                     logging.error(f'dicom_image_files = {dicom_image_files}')
#                     logging.error(f'rtss = {rtss}')
#                     output_fname_rtss = None
#             else:
#                 output_fname_rtss = None

#             # Resample to new resolution
#             if new_spacing:
#                 try:
#                     resize_nifti_files(
#                         nifti_image=output_fname,
#                         nifti_image_output=output_fname,
#                         nifti_rtss=output_fname_rtss,
#                         nifti_rtss_output=output_fname_rtss,
#                         new_spacing=new_spacing
#                     )
#                     nifti_image = nib.load(output_fname)
#                     resulting_resolution = [float(x) for x in nifti_image.header.get_zooms()]
#                     resulting_size = nifti_image.shape
#                 except Exception as e:
#                     logging.exception(f'Error resampling NIfTI files: {e}')
#                     logging.error(f'output_fname = {output_fname}')
#                     logging.error(f'output_fname_rtss = {output_fname_rtss}')

#             dicom_image_base_path = os.path.dirname(dicom_image_files[0])

#             # Collect metadata
#             metadata = {
#                 'PatientID': patient_id,
#                 'StudyDate': study_date,
#                 'Modality': modality,
#                 'nii_file': output_fname,
#                 'RTSS_nii_file': output_fname_rtss if rtss else None,
#                 'DicomStackPath': dicom_image_base_path,
#                 'DicomRTSSPath': rtss,
#                 'SOPClassUID': sop_class_uid,
#                 'OriginalResolution': original_resolution,
#                 'ResultingResolution': resulting_resolution,
#                 'OriginalSize': original_size,
#                 'ResultingSize': resulting_size,
#                 'StudyInstanceUID': study_instance_uid,
#                 'SeriesInstanceUID': series_instance_uid,
#                 'RTSS_SOPInstanceUID': rtss_sop_instance_uid,
#                 'SeriesDescription': series_description,
#                 'ProtocolName': protocol_name,
#                 'SequenceName': sequence_name,
#                 'ScanningSequence': scanning_sequence,
#             }
#             # Add additional columns from df_kwargs
#             metadata.update(df_kwargs)
#             metadata_list.append(metadata)

#             # Save metadata incrementally
#             df = pd.DataFrame([metadata])
#             if not os.path.isfile(metadata_csv_path) or overwrite:
#                 df.to_csv(metadata_csv_path, index=False)
#                 overwrite = False  # Ensure subsequent writes append to the file
#             else:
#                 df.to_csv(metadata_csv_path, mode='a', header=False, index=False)

#         else:
#             logging.error(f'No DICOM series found for: {file_dict}')
#             df = None



# Ensure any custom functions used are imported or defined elsewhere
# from your_module import (
#     pair_dicom_series_v5,
#     generate_dicom_filename,
#     convert_dicom_series_to_nifti_image,
#     convert_dicom_rtss_to_nifti,
#     correct_nifti_header,
#     resize_nifti_files,
#     get_subfolders_at_level
# )

def batch_convert_dicom_to_nifti(
    base_path: str,
    output_base_dir: str,
    structure_map: Optional[dict],
    df_kwargs: Optional[dict] = None,
    new_spacing: Optional[Tuple[float, float, float]] = None,
    folder_pattern: str = '*',
    file_pattern: str = '*dcm',
    incremental: bool = True,
    subdir_level: int = 0,
    metadata_csv_path: Optional[str] = None,
    overwrite: bool = True
):
    """
    Batch converts DICOM files to NIfTI format.

    Searches through `base_path` recursively and identifies DICOM series. Converts the DICOM series to NIfTI format
    and saves them in `output_base_dir`, along with the corresponding RTSS files. The `structure_map` is used to map
    the RTSS structures to the channels in the NIfTI files. If `structure_map` is None, all structures are included
    in the NIfTI file.

    Args:
        base_path (str): The base path where the DICOM files are located.
        output_base_dir (str): The base directory where the converted NIfTI files will be saved.
        structure_map (Optional[dict]): A dictionary mapping structure names to their corresponding values.
        df_kwargs (Optional[dict]): Additional columns to include in the metadata. 'site' must always be included.
        new_spacing (Optional[Tuple[float, float, float]]): The desired voxel spacing for the resampled NIfTI files.
        folder_pattern (str): A glob pattern to filter folders. Defaults to '*' (all folders).
        file_pattern (str): A glob pattern to filter files. Defaults to '*dcm' (all files ending with 'dcm').
        incremental (bool): If True, processes files incrementally in batches based on subdirectories.
        subdir_level (int): The directory depth level for batch processing. Defaults to 0 (top-level directory).
        metadata_csv_path (Optional[str]): Path to the CSV file containing metadata. If None, defaults to output directory.
        overwrite (bool): If False, skip processing for files that have already been converted.

    Returns:
        None
    """
    if df_kwargs is None:
        df_kwargs = {}
    # Ensure 'site' is always included in df_kwargs
    if 'site' not in df_kwargs:
        df_kwargs['site'] = None

    # Load existing metadata if CSV exists
    if metadata_csv_path is None:
        metadata_csv_path = os.path.join(output_base_dir, 'database_file.csv')
    if os.path.exists(metadata_csv_path) and not overwrite:
        existing_database_df = pd.read_csv(metadata_csv_path)
    else:
        existing_database_df = pd.DataFrame()

    top_level_path = Path(base_path)

    if incremental:
        # Get subfolders at the specified batch level using a generator
        subfolders_generator = (
            sf for sf in get_subfolders_at_level(top_level_path, subdir_level)
            if sf.match(folder_pattern)
        )

        # Convert generator to list to get total count for progress bar
        subfolders = list(subfolders_generator)
        total_subfolders = len(subfolders)

        # Add progress bar to the subfolders loop
        for subfolder in tqdm(subfolders, desc='Processing Batches', unit='subfolder', total=total_subfolders):
            # Collect files matching the file pattern within the subfolder
            files_to_process = list(subfolder.glob('**/' + file_pattern))
            if files_to_process:
                print(f"\nProcessing batch in {subfolder}")
                # Optionally, add a progress bar for collecting files
                files_to_process = list(tqdm(
                    files_to_process,
                    desc=f'Collecting files in {subfolder}',
                    unit='file'
                ))
                paired_series = pair_dicom_series_v5(
                    dicom_files=files_to_process,
                    desired_sop_class_uid="1.2.840.10008.5.1.4.1.1.4",
                    existing_database_df=existing_database_df
                )
                if len(paired_series) > 0:
                    perform_conversion(
                        paired_series,
                        output_base_dir,
                        existing_database_df,
                        metadata_csv_path,
                        structure_map,
                        df_kwargs,
                        new_spacing,
                        overwrite,
                    )
                    overwrite = False  # Ensure subsequent writes append to the file

    else:
        # Use a generator to process folders as we find them
        matching_folders = (
            p for p in top_level_path.glob('**/' + folder_pattern) if p.is_dir()
        )

        # Use tqdm to add a progress bar
        files_to_process = []
        for folder in tqdm(matching_folders, desc='Scanning Folders', unit='folder'):
            # Collect files matching the file_pattern within the folder
            files_in_folder = list(folder.glob('**/' + file_pattern))
            files_to_process.extend(files_in_folder)

        if files_to_process:
            print(f"Found {len(files_to_process)} files in folders matching '{folder_pattern}' in {base_path}")
            paired_series = pair_dicom_series_v5(
                dicom_files=files_to_process,
                desired_sop_class_uid="1.2.840.10008.5.1.4.1.1.4",
                existing_database_df=existing_database_df
            )
            if len(paired_series) > 0:
                perform_conversion(
                    paired_series,
                    output_base_dir,
                    existing_database_df,
                    metadata_csv_path,
                    structure_map,
                    df_kwargs,
                    new_spacing,
                    overwrite,
                )
                overwrite = False  # Ensure subsequent writes append to the file
        else:
            print("No files found to process.")

def perform_conversion(
    paired_series: List[Dict[str, Any]],
    output_base_dir: str,
    existing_database_df: pd.DataFrame,
    metadata_csv_path: Optional[str],
    structure_map: Optional[dict],
    df_kwargs: Optional[dict] = None,
    new_spacing: Optional[Tuple[float, float, float]] = None,
    overwrite: bool = True
):
    """
    Performs the conversion of paired DICOM series to NIfTI format and saves the metadata.

    Args:
        paired_series (List[Dict[str, Any]]): A list of dictionaries, each containing 'image' and 'rtss' keys
            pointing to the DICOM image files and RTSS files respectively.
        output_base_dir (str): The base directory where the converted NIfTI files will be saved.
        existing_database_df (pd.DataFrame): DataFrame containing existing metadata, used to avoid duplicate processing.
        metadata_csv_path (Optional[str]): Path to the CSV file containing metadata. If None, defaults to output directory.
        structure_map (Optional[dict]): A dictionary mapping structure names to their corresponding values.
        df_kwargs (Optional[dict]): Additional columns to include in the metadata. 'site' must always be included.
        new_spacing (Optional[Tuple[float, float, float]]): The desired voxel spacing for the resampled NIfTI files.
        overwrite (bool): If False, skip processing for files that have already been converted.

    Returns:
        None
    """
    if df_kwargs is None:
        df_kwargs = {}
    metadata_list = []

    for file_dict in tqdm(paired_series, desc='Converting DICOM to NIfTI'):
        dicom_image_files = file_dict['image']
        rtss = file_dict.get('rtss')
        logging.info(f'dicom file paths = {dicom_image_files[0]}')
        logging.info(f'rtss file = {rtss}')

        if dicom_image_files:
            dicom_file_path = dicom_image_files[0]
            fname = generate_dicom_filename(dicom_file_path)
            output_dir = os.path.join(output_base_dir, fname)
            output_fname = os.path.join(output_dir, fname + '.nii.gz')

            # Skip processing if this series is already in the metadata CSV
            if not overwrite and not existing_database_df.empty and existing_database_df['nii_file'].str.contains(output_fname).any():
                logging.info(f'Skipping already converted series: {output_fname}')
                continue

            try:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                convert_dicom_series_to_nifti_image(dicom_image_files, output_fname)

                # Extract DICOM metadata
                dicom_metadata = dcmread(dicom_file_path)
                patient_id = getattr(dicom_metadata, 'PatientID', None)
                study_date = getattr(dicom_metadata, 'StudyDate', None)
                modality = getattr(dicom_metadata, 'Modality', None)
                sop_class_uid = getattr(dicom_metadata, 'SOPClassUID', None)
                pixel_spacing = getattr(dicom_metadata, 'PixelSpacing', [None, None])
                slice_thickness = getattr(dicom_metadata, 'SliceThickness', None)
                original_resolution = [float(x) if x is not None else None for x in pixel_spacing] + [float(slice_thickness) if slice_thickness is not None else None]
                study_instance_uid = getattr(dicom_metadata, 'StudyInstanceUID', None)
                series_instance_uid = getattr(dicom_metadata, 'SeriesInstanceUID', None)
                series_description = getattr(dicom_metadata, 'SeriesDescription', None)
                protocol_name = getattr(dicom_metadata, 'ProtocolName', None)
                sequence_name = getattr(dicom_metadata, 'SequenceName', None)
                scanning_sequence = getattr(dicom_metadata, 'ScanningSequence', None)

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
                continue

            output_fname_rtss = None
            if rtss:
                try:
                    output_fname_rtss = os.path.join(output_dir, fname + '_RTSS.nii.gz')
                    convert_dicom_rtss_to_nifti(dicom_image_files, rtss, output_fname_rtss, structure_map=structure_map)

                    # Fix NIfTI RTSS headers and affine matrix
                    src = output_fname
                    dst = output_fname_rtss
                    new_nifti_rtss = correct_nifti_header(src, dst)
                    nib.save(new_nifti_rtss, dst)
                except Exception as e:
                    logging.exception(f'Error converting RTSS to NIfTI: {e}')
                    logging.error(f'dicom_image_files = {dicom_image_files}')
                    logging.error(f'rtss = {rtss}')
                    output_fname_rtss = None

            # Resample to new resolution
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

            dicom_image_base_path = os.path.dirname(dicom_image_files[0])

            # Collect metadata
            metadata = {
                'PatientID': patient_id,
                'StudyDate': study_date,
                'Modality': modality,
                'nii_file': output_fname,
                'RTSS_nii_file': output_fname_rtss,
                'DicomStackPath': dicom_image_base_path,
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
                'ProtocolName': protocol_name,
                'SequenceName': sequence_name,
                'ScanningSequence': scanning_sequence,
            }
            # Add additional columns from df_kwargs
            metadata.update(df_kwargs)
            metadata_list.append(metadata)

            # Save metadata incrementally
            df = pd.DataFrame([metadata])
            if not os.path.isfile(metadata_csv_path) or overwrite:
                df.to_csv(metadata_csv_path, index=False)
                overwrite = False # Ensure subsequent writes append to the file
            else:
                df.to_csv(metadata_csv_path, mode='a', header=False, index=False)

        else:
            logging.error(f'No DICOM series found for: {file_dict}')


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