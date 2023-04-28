# 

Package to perform conversions between dicom and nifti file formats for medical images. It relies mainly on SimpleITK, but can also handle nibabel files to some extent. It also contains simple visualization functions. It doesn't do anything new, but is meant as a convenient package for some functions that I end up using often and want to treat in a consistent manner.

# To do:
- implement pydicom version of all functions
    - [x] load_dicom_images_pydicom (and support functions)
    - [] save_itk_image_as_nifti_sitk
    - [] convert_dicom_to_nifti (also add flag to choose sitk or pydicom, with one chosen as default)
    - [] others?
- test to see that you get the same results in terms of nifti files.
- add tests to key functions